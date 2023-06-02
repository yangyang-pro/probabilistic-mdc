import itertools
import networkx as nx
import numpy as np

import torch
import torchvision

from data import TabularDataLoader, IMGDataLoader
from mpe import compute_mpe
from utils import subset, powerset
from utils import train_tabular_base_clfs, train_img_base_clfs
from utils import compute_cll_tabular_clfs, compute_cll_img_clfs
from utils import predict_proba_base_clfs, predict_log_proba_base_clfs
from temperature_scaling import ModelWithTemperature

from tqdm import tqdm

from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference.ExactInference import VariableElimination

from pygobnilp.gobnilp import Gobnilp


class TabularClassifier:
    def __init__(self,
                 palim: int = 3,
                 disclim: int = 5,
                 base: str = None):
        self.palim = palim
        self.base = base
        self.disc_lim = disclim

        self.label_domains = None
        self.discrete_feature_domains = None
        self.selected_discrete_features = None
        self.classifiers = None
        self.parent_dict = None
        self.scores = None
        self.__best_subset_scores = None
        self.bn = None

    def __init_structures(self, label_domains, discrete_feature_domains):
        self.label_domains = label_domains
        self.discrete_feature_domains = discrete_feature_domains
        if discrete_feature_domains:
            self.selected_discrete_features = {node: {} for node in list(label_domains.keys())}
        self.classifiers = {node: {} for node in list(label_domains.keys())}
        # A dictionary that scores[i][parents] stores the CLL score for node i given its parents
        self.scores = {node: {} for node in list(label_domains.keys())}
        self.__best_subset_scores = {node: {} for node in list(label_domains.keys())}
        self.bn = None

    def fit(self, train_data_loader: TabularDataLoader, show_log=False):
        self.__init_structures(train_data_loader.label_domains, train_data_loader.discrete_feature_domains)
        all_nodes = list(self.label_domains.keys())
        candidate_child_nodes = list(self.label_domains.keys())
        stopped_parents = {node: [] for node in candidate_child_nodes}
        for n_parents in tqdm(range(self.palim + 1)):
            all_parents = {node: list(subset(list(set(all_nodes) - {node}), n_elements=n_parents))
                           for node in candidate_child_nodes}
            for node in list(all_parents.keys()):
                cur_parents = all_parents[node]
                is_stopped_cur_parents = [False for _ in range(len(cur_parents))]
                for parents_idx, parents in enumerate(cur_parents):
                    parent_set = frozenset(parents)
                    parent_domains = [self.label_domains[parent_id] for parent_id in parents]
                    configurations = list(itertools.product(*parent_domains))
                    penalty = -0.5 * np.log(len(train_data_loader.continuous_features)) \
                              * len(configurations) * (len(self.label_domains[node]) - 1)
                    if show_log:
                        print('{:<10}{:<5}'
                              '{:<10}{:<30}'
                              '{:<20}{:<20.3f}'.format('Node:', node,
                                                       'Parents:', '{}'.format(list(parents)),
                                                       'Penalty:', penalty))
                    if len(parents) != 0:
                        parent_subsets = list(subset(parents, len(parents) - 1))
                        if not set(parent_subsets).isdisjoint(set(stopped_parents[node])):
                            stopped_parents[node].append(parents)
                            is_stopped_cur_parents[parents_idx] = True
                            continue
                        subset_scores = [self.__best_subset_scores[node][frozenset(parent_subset)]
                                         for parent_subset in parent_subsets]
                        best_subset_score = max(subset_scores)
                        if best_subset_score >= penalty:
                            self.__best_subset_scores[node][parent_set] = best_subset_score
                            stopped_parents[node].append(parents)
                            is_stopped_cur_parents[parents_idx] = True
                            if show_log:
                                print('{:<10}{:<5}'
                                      '{:<10}{:<30}'
                                      '{}'.format('Node:', node,
                                                  'Parents:', '{}'.format(list(parents)),
                                                  'Pruned and stop growing!'))
                            continue
                    config_clfs = dict()
                    bic_score = penalty
                    for configuration in configurations:
                        configuration = tuple(configuration)
                        local_features, local_labels = train_data_loader.get_slices(node, parents, configuration)
                        if local_features.shape[0] == 0:
                            clf = [1 / len(self.label_domains[node])] * len(self.label_domains[node])
                            config_clfs[configuration] = clf
                            continue
                        if np.all(local_labels == local_labels[0]):
                            clf = [0] * len(self.label_domains[node])
                            clf[int(local_labels[0])] = 1
                            config_clfs[configuration] = clf
                            continue
                        else:
                            clf = self._create_base_clfs(self.base)
                        if isinstance(clf, torch.nn.Module):
                            x_train, y_train = torch.Tensor(local_features), torch.Tensor(local_labels)
                            y_train = y_train.type(torch.LongTensor)
                            train_tabular_base_clfs(model=clf, x_train=x_train, y_train=y_train)
                        elif isinstance(clf, (ClassifierMixin, list)):
                            x_train, y_train = local_features, local_labels
                            train_tabular_base_clfs(model=clf, x_train=x_train, y_train=y_train)
                        else:
                            raise NotImplementedError
                        config_clfs[configuration] = clf
                        bic_score += compute_cll_tabular_clfs(model=clf, x_train=x_train, y_train=y_train)
                    if show_log:
                        print('{:<10}{:<5}'
                              '{:<10}{:<30}'
                              '{:<20}{:<20.3f}'.format('Node:', node,
                                                       'Parents:', '{}'.format(list(parents)),
                                                       'CLL Score:', bic_score))
                    self.__best_subset_scores[node][parent_set] = bic_score
                    if len(parents) != 0:
                        parent_subsets = list(subset(parents, len(parents) - 1))
                        subset_scores = [self.__best_subset_scores[node][frozenset(parent_subset)]
                                         for parent_subset in parent_subsets]
                        best_subset_score = max(subset_scores)
                        if best_subset_score >= bic_score:
                            self.__best_subset_scores[node][parent_set] = best_subset_score
                            if show_log:
                                print('{:<10}{:<5}'
                                      '{:<10}{:<30}'
                                      '{}'.format('Node:', node,
                                                  'Parents:', '{}'.format(list(parents)),
                                                  'Pruned!'))
                            continue
                    self.classifiers[node][parent_set] = config_clfs
                    self.scores[node][parent_set] = bic_score
                if np.all(is_stopped_cur_parents):
                    if show_log:
                        print('{:<10}{:<5}{}'.format('Node:', node, 'Stopped growing!'))
                    candidate_child_nodes.remove(node)

    def fit_mix(self, train_data_loader: TabularDataLoader, show_log=False):
        self.__init_structures(train_data_loader.label_domains, train_data_loader.discrete_feature_domains)
        all_nodes = list(self.label_domains.keys())
        cand_child_nodes = list(self.label_domains.keys())
        stopped_parents = {node: [] for node in all_nodes}
        all_discs = list(self.discrete_feature_domains.keys())
        for n_parents in range(self.palim + 1):
            all_parents = {node: list(subset(list(set(all_nodes) - {node}), n_elements=n_parents))
                           for node in cand_child_nodes}
            for node in list(all_parents.keys()):
                cur_parents = all_parents[node]
                is_stopped_cur_parents = [False for _ in range(len(cur_parents))]
                for parents_idx, parents in enumerate(cur_parents):
                    parent_set = frozenset(parents)

                    best_subset_score = -np.inf
                    if len(parents) > 0:
                        parent_subsets = list(subset(parents, len(parents) - 1))
                        if not set(parent_subsets).isdisjoint(set(stopped_parents[node])):
                            stopped_parents[node].append(parents)
                            is_stopped_cur_parents[parents_idx] = True
                            continue
                        subset_scores = [self.__best_subset_scores[node][frozenset(parent_subset)]
                                         for parent_subset in parent_subsets]
                        best_subset_score = max(subset_scores)

                    is_stopped_parents_target = True

                    best_base_clfs = dict()
                    best_parents_disc = None
                    best_cll_score = -np.inf
                    stopped_discs = []
                    best_sub_disc_scores = dict()
                    for n_parents_disc in range(self.disc_lim + 1):
                        all_parents_disc = list(subset(all_discs, n_elements=n_parents_disc))
                        is_stopped_cur_discs = True
                        for parents_disc_idx, parents_disc in enumerate(tqdm(all_parents_disc)):
                            parents_disc_target = parents + parents_disc
                            parent_target_domains = [self.label_domains[parent] for parent in parents]
                            parent_disc_domains = [self.discrete_feature_domains[parent] for parent in parents_disc]
                            parent_domains = parent_target_domains + parent_disc_domains
                            # disc_configurations = list(itertools.product(*parent_disc_domains))
                            configurations = list(itertools.product(*parent_domains))
                            # penalty_disc = -0.5 * np.log(len(train_data_loader.continuous_features)) \
                            #                * len(disc_configurations) * (len(self.label_domains[node]) - 1)
                            penalty_disc = -0.5 * np.log(len(train_data_loader.continuous_features)) * \
                                           len(configurations) * (len(self.label_domains[node]) - 1)

                            if show_log:
                                print('{:<10}{:<5}'
                                      '{:<10}{:<30}'
                                      '{:<20}{:<30}'
                                      '{:<20}{:<20.3f}'.format('Node:', node,
                                                               'Parents:', '{}'.format(list(parents)),
                                                               'Discrete: ', '{}'.format(list(parents_disc)),
                                                               'Penalty:', penalty_disc))
                            if len(parents) > 0:
                                if best_subset_score >= penalty_disc:
                                    continue
                            is_stopped_parents_target = False

                            best_sub_disc_score = None
                            if len(parents_disc) > 0:
                                disc_subsets = list(subset(parents_disc, len(parents_disc) - 1))
                                if not set(disc_subsets).isdisjoint(set(stopped_discs)):
                                    stopped_discs.append(parents_disc)
                                    if show_log:
                                        print('{:<10}{:<5}'
                                              '{:<10}{:<30}'
                                              '{:<20}{:<30}'
                                              '{}'.format('Node:', node,
                                                          'Parents:', '{}'.format(list(parents)),
                                                          'Discrete: ', '{}'.format(list(parents_disc)),
                                                          'Pruned and stop growing!'))
                                    continue
                                sub_disc_scores = [best_sub_disc_scores[frozenset(disc_subset)]
                                                   for disc_subset in disc_subsets]
                                best_sub_disc_score = max(sub_disc_scores)
                                if best_sub_disc_score >= penalty_disc:
                                    best_sub_disc_scores[parent_set] = best_sub_disc_score
                                    stopped_discs.append(parents_disc)
                                    if show_log:
                                        print('{:<10}{:<5}'
                                              '{:<10}{:<30}'
                                              '{:<20}{:<30}'
                                              '{}'.format('Node:', node,
                                                          'Parents:', '{}'.format(list(parents)),
                                                          'Discrete:', '{}'.format(list(parents_disc)),
                                                          'Pruned and stop growing!'))
                                    continue
                            is_stopped_cur_discs = False

                            cll_score = penalty_disc
                            base_clfs = dict()
                            for configuration in configurations:
                                configuration = tuple(configuration)
                                local_features, local_labels = train_data_loader.get_slices(node,
                                                                                            parents_disc_target,
                                                                                            configuration)
                                if local_features.shape[0] == 0:
                                    clf = [1 / len(self.label_domains[node])] * len(self.label_domains[node])
                                # strange cases occur in some data sets that two samples with the same
                                # feature vector have different labels
                                elif np.all(local_features == local_features[0]) and \
                                    not np.all(local_labels == local_labels[0]):
                                    clf = [1 / len(self.label_domains[node])] * len(self.label_domains[node])
                                elif np.all(local_labels == local_labels[0]):
                                    clf = [0] * len(self.label_domains[node])
                                    clf[int(local_labels[0])] = 1
                                else:
                                    clf = self._create_base_clfs(self.base)
                                    x_train, y_train = local_features, local_labels
                                    train_tabular_base_clfs(model=clf, x_train=x_train, y_train=y_train)
                                    cll_score += compute_cll_tabular_clfs(model=clf, x_train=x_train, y_train=y_train)
                                base_clfs[configuration] = clf

                            if show_log:
                                print('{:<10}{:<5}'
                                      '{:<10}{:<30}'
                                      '{:<20}{:<30}'
                                      '{:<20}{:<20.3f}'.format('Node:', node,
                                                               'Parents:', '{}'.format(list(parents)),
                                                               'Discrete: ', '{}'.format(list(parents_disc)),
                                                               'CLL Score:', cll_score))
                            if len(parents_disc) > 0:
                                if best_sub_disc_score >= penalty_disc:
                                    best_sub_disc_scores[parent_set] = best_sub_disc_score
                                    if show_log:
                                        print('{:<10}{:<5}'
                                              '{:<10}{:<30}'
                                              '{:<20}{:<30}'
                                              '{}'.format('Node:', node,
                                                          'Parents:', '{}'.format(list(parents)),
                                                          'Discrete:', '{}'.format(list(parents_disc)),
                                                          'Pruned and stop growing!'))
                                    continue

                            best_sub_disc_scores[frozenset(parents_disc)] = cll_score
                            if cll_score > best_cll_score:
                                best_base_clfs.update(base_clfs)
                                best_parents_disc = parents_disc
                                best_cll_score = cll_score
                        if is_stopped_cur_discs:
                            if show_log:
                                print('{:<10}{:<5}{}'.format('Node:', node, 'Discrete stopped growing!'))
                            break

                    if is_stopped_parents_target:
                        self.__best_subset_scores[node][parent_set] = best_subset_score
                        stopped_parents[node].append(parents)
                        is_stopped_cur_parents[parents_idx] = True
                        if show_log:
                            print('{:<10}{:<5}'
                                  '{:<10}{:<30}'
                                  '{}'.format('Node:', node,
                                              'Parents:', '{}'.format(list(parents)),
                                              'Pruned and stop growing!'))
                        continue

                    if len(parents) != 0:
                        parent_subsets = list(subset(parents, len(parents) - 1))
                        subset_scores = [self.__best_subset_scores[node][frozenset(parent_subset)]
                                         for parent_subset in parent_subsets]
                        best_subset_score = max(subset_scores)
                        if best_subset_score >= best_cll_score:
                            self.__best_subset_scores[node][parent_set] = best_subset_score
                            if show_log:
                                print('{:<10}{:<5}'
                                      '{:<10}{:<30}'
                                      '{}'.format('Node:', node,
                                                  'Parents:', '{}'.format(list(parents)),
                                                  'Pruned!'))
                            continue

                    self.__best_subset_scores[node][parent_set] = best_cll_score
                    self.classifiers[node][parent_set] = best_base_clfs
                    self.selected_discrete_features[node][parent_set] = best_parents_disc
                    self.scores[node][parent_set] = best_cll_score
                    if show_log:
                        print('{:<10}{:<5}'
                              '{:<10}{:<30}'
                              '{:<20}{:<30}'
                              '{:<20}{:<20.3f}'.format('Node:', node,
                                                       'Parents:', '{}'.format(list(parents)),
                                                       'Selected: ', '{}'.format(list(best_parents_disc)),
                                                       'CLL Score:', self.__best_subset_scores[node][parent_set]))
                if np.all(is_stopped_cur_parents):
                    if show_log:
                        print('{:<10}{:<5}{}'.format('Node:', node, 'Stopped growing!'))
                    cand_child_nodes.remove(node)

    def learn_structure(self, plot=True):
        m = Gobnilp()
        m.learn(local_scores_source=self.scores, palim=self.palim, plot=plot)
        self.bn = m.learned_bn
        parent_dict = {node: [] for node in list(self.label_domains.keys())}
        for edge in list(self.bn.edges):
            parent_dict[edge[1]].append(edge[0])
        for k, v in parent_dict.items():
            parent_dict[k] = sorted(v)
        self.parent_dict = parent_dict
        # Prune the clfs and selected discrete features dicts based on the learned structure
        for node in list(self.label_domains.keys()):
            for parent_set in list(self.classifiers[node].keys()):
                if frozenset(parent_dict[node]) != parent_set:
                    self.classifiers[node].pop(parent_set, None)
        if self.selected_discrete_features:
            for node in list(self.label_domains.keys()):
                for parent_set in list(self.selected_discrete_features[node].keys()):
                    if frozenset(parent_dict[node]) != parent_set:
                        self.selected_discrete_features[node].pop(parent_set, None)

    def inference(self, test_features):
        bn = BayesianNetwork()
        bn.add_nodes_from(self.bn.nodes)
        bn.add_edges_from(self.bn.edges)
        pred_labels = np.empty(shape=(len(test_features), len(self.label_domains)))
        for i, test_feature in enumerate(tqdm(test_features)):
            for child, parents in self.parent_dict.items():
                parent_set = frozenset(parents)
                parent_domains = [self.label_domains[parent_id] for parent_id in parents]
                configurations = list(itertools.product(*parent_domains))
                cp_matrix = np.empty(shape=(len(self.label_domains[child]), len(configurations)), dtype=np.float32)
                for j, configuration in enumerate(configurations):
                    clf = self.classifiers[child][parent_set][tuple(configuration)]
                    cond_probs = predict_proba_base_clfs(clf, test_feature)
                    if isinstance(clf, ClassifierMixin) and len(self.label_domains[child]) > len(clf.classes_):
                        corrected_cond_probs = np.zeros(shape=len(self.label_domains[child]), dtype=np.float32)
                        for idx_c, c in enumerate(clf.classes_):
                            corrected_cond_probs[c] = cond_probs[:, idx_c]
                        cond_probs = corrected_cond_probs
                    cp_matrix[:, j] = cond_probs
                cpd = TabularCPD(variable=child, variable_card=len(self.label_domains[child]), values=cp_matrix,
                                 evidence=parents,
                                 evidence_card=[len(self.label_domains[parent]) for parent in parents])
                bn.add_cpds(cpd)
            max_prob, assignments = compute_mpe(graph=bn,
                                                variables=['0'],
                                                evidence=None,
                                                elimination_order=None,
                                                show_progress=False)
            for v in assignments.keys():
                pred_labels[i, int(v)] = assignments[v]
            bn.cpds = []
        return pred_labels

    def inference_hamming(self, test_features):
        bn = BayesianNetwork()
        bn.add_nodes_from(self.bn.nodes)
        bn.add_edges_from(self.bn.edges)
        pred_labels = np.empty(shape=(len(test_features), len(self.label_domains)))
        for i, test_feature in enumerate(tqdm(test_features)):
            for child, parents in self.parent_dict.items():
                parent_set = frozenset(parents)
                parent_domains = [self.label_domains[parent_id] for parent_id in parents]
                configurations = list(itertools.product(*parent_domains))
                cp_matrix = np.empty(shape=(len(self.label_domains[child]), len(configurations)), dtype=np.float32)
                for j, configuration in enumerate(configurations):
                    clf = self.classifiers[child][parent_set][tuple(configuration)]
                    cond_probs = predict_proba_base_clfs(clf, test_feature)
                    if isinstance(clf, ClassifierMixin) and len(self.label_domains[child]) > len(clf.classes_):
                        corrected_cond_probs = np.zeros(shape=len(self.label_domains[child]), dtype=np.float32)
                        for idx_c, c in enumerate(clf.classes_):
                            corrected_cond_probs[c] = cond_probs[:, idx_c]
                        cond_probs = corrected_cond_probs
                    cp_matrix[:, j] = cond_probs
                cpd = TabularCPD(variable=child, variable_card=len(self.label_domains[child]), values=cp_matrix,
                                 evidence=parents,
                                 evidence_card=[len(self.label_domains[parent]) for parent in parents])
                bn.add_cpds(cpd)
            inferencer = VariableElimination(bn)
            for k in self.label_domains.keys():
                pred_labels[i, int(k)] = inferencer.map_query(variables=[k], show_progress=False)[k]
            bn.cpds = []
        return pred_labels

    def inference_mix(self, cont_features, disc_features):
        bn = BayesianNetwork(self.bn)
        pred_labels = np.empty(shape=(len(cont_features), len(self.label_domains)))
        for i, cont_feature in enumerate(tqdm(cont_features)):
            disc_feature = disc_features[i]
            for child, parents in self.parent_dict.items():
                parent_set = frozenset(parents)
                parents_disc = self.selected_discrete_features[child][parent_set]
                parent_disc_indices = [ord(c) - 97 for c in parents_disc]
                selected_disc_values = disc_feature[parent_disc_indices]
                parent_domains = [self.label_domains[parent_id] for parent_id in parents]
                target_configurations = list(itertools.product(*parent_domains))
                cp = np.empty(shape=(len(self.label_domains[child]), len(target_configurations)), dtype=np.float32)
                for j, target_configuration in enumerate(target_configurations):
                    configuration = target_configuration + tuple(selected_disc_values)
                    clf = self.classifiers[child][parent_set][tuple(configuration)]
                    cond_probs = predict_proba_base_clfs(clf, cont_feature)
                    if isinstance(clf, ClassifierMixin) and len(self.label_domains[child]) > len(clf.classes_):
                        corrected_cond_probs = np.zeros(shape=len(self.label_domains[child]), dtype=np.float32)
                        for idx_c, c in enumerate(clf.classes_):
                            corrected_cond_probs[c] = cond_probs[:, idx_c]
                        cond_probs = corrected_cond_probs
                    cp[:, j] = cond_probs
                cpd = TabularCPD(variable=child, variable_card=len(self.label_domains[child]), values=cp,
                                 evidence=parents,
                                 evidence_card=[len(self.label_domains[parent]) for parent in parents])
                bn.add_cpds(cpd)
            max_prob, assignments = compute_mpe(graph=bn,
                                                variables=['0'],
                                                evidence=None,
                                                elimination_order=None,
                                                show_progress=False)
            for v in assignments.keys():
                pred_labels[i, int(v)] = assignments[v]
            bn.cpds = []
        return pred_labels

    def inference_hamming_mix(self, cont_features, disc_features):
        bn = BayesianNetwork(self.bn)
        pred_labels = np.empty(shape=(len(cont_features), len(self.label_domains)))
        for i, cont_feature in enumerate(tqdm(cont_features)):
            disc_feature = disc_features[i]
            for child, parents in self.parent_dict.items():
                parent_set = frozenset(parents)
                parents_disc = self.selected_discrete_features[child][parent_set]
                parent_disc_indices = [ord(c) - 97 for c in parents_disc]
                selected_disc_values = disc_feature[parent_disc_indices]
                parent_domains = [self.label_domains[parent_id] for parent_id in parents]
                target_configurations = list(itertools.product(*parent_domains))
                cp = np.empty(shape=(len(self.label_domains[child]), len(target_configurations)), dtype=np.float32)
                for j, target_configuration in enumerate(target_configurations):
                    configuration = target_configuration + tuple(selected_disc_values)
                    clf = self.classifiers[child][parent_set][tuple(configuration)]
                    cond_probs = predict_proba_base_clfs(clf, cont_feature)
                    if isinstance(clf, ClassifierMixin) and len(self.label_domains[child]) > len(clf.classes_):
                        corrected_cond_probs = np.zeros(shape=len(self.label_domains[child]), dtype=np.float32)
                        for idx_c, c in enumerate(clf.classes_):
                            corrected_cond_probs[c] = cond_probs[:, idx_c]
                        cond_probs = corrected_cond_probs
                    cp[:, j] = cond_probs
                cpd = TabularCPD(variable=child, variable_card=len(self.label_domains[child]), values=cp,
                                 evidence=parents,
                                 evidence_card=[len(self.label_domains[parent]) for parent in parents])
                bn.add_cpds(cpd)
            inferencer = VariableElimination(bn)
            for k in self.label_domains.keys():
                pred_labels[i, int(k)] = inferencer.map_query(variables=[k], show_progress=False)[k]
            bn.cpds = []
        return pred_labels

    def inference_seq(self, test_features):
        pred_labels = np.zeros(shape=(len(test_features), len(self.label_domains)), dtype=int)
        params = {node: {} for node in list(self.label_domains.keys())}
        for child, parents in self.parent_dict.items():
            parent_set = frozenset(parents)
            parent_domains = [self.label_domains[parent_id] for parent_id in parents]
            configurations = list(itertools.product(*parent_domains))
            log_probs = np.empty(shape=(len(test_features), len(self.label_domains[child]), len(configurations)),
                                 dtype=np.float32)
            for j, configuration in enumerate(configurations):
                clf = self.classifiers[child][parent_set][tuple(configuration)]
                log_prob = predict_proba_base_clfs(clf, test_features)
                if isinstance(clf, ClassifierMixin) and len(self.label_domains[child]) > len(clf.classes_):
                    corrected_log_prob = np.zeros(shape=(len(test_features), len(self.label_domains[child])),
                                                  dtype=np.float32)
                    for idx_c, c in enumerate(clf.classes_):
                        corrected_log_prob[:, c] = log_prob[:, idx_c]
                    log_prob = corrected_log_prob
                log_probs[:, :, j] = log_prob
            params[child][parent_set] = log_probs

        bfo = list(nx.topological_sort(self.bn))
        root = bfo[0]
        params_root = params[root][frozenset()]
        assignment_root = np.argmax(params_root, axis=1).squeeze()
        pred_labels[:, int(root)] = assignment_root
        for node in bfo[1:]:
            parents = self.parent_dict[node]
            parent_set = frozenset(parents)
            parent_indices = [int(parent) for parent in parents]
            params_node_all_configs = params[node][parent_set]
            n_features, n_node_states, n_configs = params_node_all_configs.shape
            parent_domains = [self.label_domains[parent_id] for parent_id in parents]
            configurations = np.array(list(itertools.product(*parent_domains)))
            parent_assignments = pred_labels[:, parent_indices]

            # https://stackoverflow.com/questions/64930665/find-indices-of-rows-of-numpy-2d-array-in-another-2d-array
            indices = parent_assignments == configurations[:, None]
            positions = np.zeros(n_features, dtype=int)
            p, q = np.where(np.all(indices, axis=2) == True)
            positions[q] = p

            params_node = params_node_all_configs[np.arange(n_features), :, positions]
            assignment_node = np.argmax(params_node, axis=1).squeeze()
            pred_labels[:, int(node)] = assignment_node
        return pred_labels

    @staticmethod
    def _create_base_clfs(base):
        if base == 'lr':
            return SGDClassifier(loss='log_loss', n_jobs=-1)
        elif base == 'nb':
            return GaussianNB()
        elif base == 'rf':
            return RandomForestClassifier(n_jobs=-1)
        elif base == 'svm':
            return SVC(probability=True)
        else:
            raise NotImplementedError


class IMGClassifier:
    def __init__(self,
                 palim: int = 3):
        self.palim = palim

        self.label_domains = None
        self.classifiers = None
        self.parent_dict = None
        self.scores = None
        self.__best_subset_scores = None
        self.bn = None

    def __init_structures(self, label_domains):
        self.label_domains = label_domains
        self.classifiers = {node: {} for node in list(label_domains.keys())}
        self.scores = {node: {} for node in list(label_domains.keys())}
        self.__best_subset_scores = {node: {} for node in list(label_domains.keys())}

    def fit(self,
            train_data_loader: IMGDataLoader,
            batch_size=32,
            learning_rate=0.01,
            n_epochs=5,
            show_log=False,
            base_trainer=None,
            calibration=True):
        self.__init_structures(train_data_loader.label_domains)
        all_nodes = list(self.label_domains.keys())
        candidate_child_nodes = list(self.label_domains.keys())
        stopped_parents = {node: [] for node in candidate_child_nodes}
        for n_parents in tqdm(range(self.palim + 1)):
            all_parents = {node: list(subset(list(set(all_nodes) - {node}), n_elements=n_parents))
                           for node in candidate_child_nodes}
            for node in list(all_parents.keys()):
                cur_parents = all_parents[node]
                is_stopped_cur_parents = True
                for parents_idx, parents in enumerate(cur_parents):
                    parent_set = frozenset(parents)
                    parent_domains = [self.label_domains[parent_id] for parent_id in parents]
                    configurations = list(itertools.product(*parent_domains))
                    penalty = -0.5 * np.log(len(train_data_loader.labels)) \
                              * len(configurations) * (len(self.label_domains[node]) - 1)
                    if show_log:
                        print('{:<10}{:<5}'
                              '{:<10}{:<30}'
                              '{:<20}{:<20.3f}'.format('Node:', node,
                                                       'Parents:', '{}'.format(list(parents)),
                                                       'Penalty:', penalty))
                    best_subset_score = -np.inf
                    if len(parents) != 0:
                        parent_subsets = list(subset(parents, len(parents) - 1))
                        if not set(parent_subsets).isdisjoint(set(stopped_parents[node])):
                            stopped_parents[node].append(parents)
                            continue
                        subset_scores = [self.__best_subset_scores[node][frozenset(parent_subset)]
                                         for parent_subset in parent_subsets]
                        best_subset_score = max(subset_scores)
                        if best_subset_score >= penalty:
                            self.__best_subset_scores[node][parent_set] = best_subset_score
                            stopped_parents[node].append(parents)
                            if show_log:
                                print('{:<10}{:<5}'
                                      '{:<10}{:<30}'
                                      '{}'.format('Node:', node,
                                                  'Parents:', '{}'.format(list(parents)),
                                                  'Pruned and stop growing!'))
                            continue
                    is_stopped_cur_parents = False

                    config_clfs = dict()
                    bic_score = penalty
                    for configuration in configurations:
                        configuration = tuple(configuration)
                        local_files, local_labels = train_data_loader.get_slices(node, parents, configuration)
                        local_features, local_labels = train_data_loader.get_slices(node, parents, configuration)
                        if len(local_labels) == 0:
                            clf = [1 / len(self.label_domains[node])] * len(self.label_domains[node])
                            config_clfs[configuration] = clf
                            continue
                        if np.all(local_labels == local_labels[0]):
                            clf = [0] * len(self.label_domains[node])
                            clf[int(local_labels[0])] = 1
                            config_clfs[configuration] = clf
                            continue
                        else:
                            clf = self.__create_base_clfs(output_dim=len(self.label_domains[node]))
                        local_data_loader = train_data_loader.create_local_data_loaders(local_files=local_files,
                                                                                        local_labels=local_labels,
                                                                                        batch_size=batch_size)

                        if base_trainer:
                            base_trainer(model=clf,
                                         train_data_loader=local_data_loader,
                                         n_classes=len(self.label_domains[node]),
                                         learning_rate=learning_rate,
                                         n_epochs=n_epochs)
                        else:
                            train_img_base_clfs(model=clf,
                                                train_data_loader=local_data_loader,
                                                learning_rate=learning_rate,
                                                n_epochs=n_epochs)
                        if calibration:
                            clf = ModelWithTemperature(clf)
                            clf.set_temperature(local_data_loader)
                        config_clfs[configuration] = clf
                        bic_score += compute_cll_img_clfs(model=clf, train_data_loader=local_data_loader)
                    if show_log:
                        print('{:<10}{:<5}'
                              '{:<10}{:<30}'
                              '{:<20}{:<20.3f}'.format('Node:', node,
                                                       'Parents:', '{}'.format(list(parents)),
                                                       'CLL Score:', bic_score))

                    if len(parents) != 0:
                        if best_subset_score >= bic_score:
                            self.__best_subset_scores[node][parent_set] = best_subset_score
                            if show_log:
                                print('{:<10}{:<5}'
                                      '{:<10}{:<30}'
                                      '{}'.format('Node:', node,
                                                  'Parents:', '{}'.format(list(parents)),
                                                  'Pruned!'))
                            continue

                    self.__best_subset_scores[node][parent_set] = bic_score
                    self.classifiers[node][parent_set] = config_clfs
                    self.scores[node][parent_set] = bic_score
                if is_stopped_cur_parents:
                    if show_log:
                        print('{:<10}{:<5}{}'.format('Node:', node, 'Stopped growing!'))
                    candidate_child_nodes.remove(node)

    def learn_structure(self, plot=True):
        m = Gobnilp()
        m.learn(local_scores_source=self.scores, palim=self.palim, plot=plot)
        self.bn = m.learned_bn
        parent_dict = {node: [] for node in list(self.label_domains.keys())}
        for edge in list(self.bn.edges):
            parent_dict[edge[1]].append(edge[0])
        for k, v in parent_dict.items():
            parent_dict[k] = sorted(v)
        self.parent_dict = parent_dict
        # Prune the clfs and selected discrete features dicts based on the learned structure
        for node in list(self.label_domains.keys()):
            for parent_set in list(self.classifiers[node].keys()):
                if frozenset(parent_dict[node]) != parent_set:
                    self.classifiers[node].pop(parent_set, None)

    def inference(self, test_dataloader: IMGDataLoader):
        bn = BayesianNetwork()
        bn.add_nodes_from(self.bn.nodes)
        bn.add_edges_from(self.bn.edges)
        pred_labels = np.empty(shape=(len(test_dataloader.labels), len(self.label_domains)))
        test_img_dataset = test_dataloader.create_img_datasets()
        for i, (test_img, _) in enumerate(tqdm(test_img_dataset)):
            test_img = test_img.unsqueeze(dim=0)
            for child, parents in self.parent_dict.items():
                parent_domains = [self.label_domains[parent_id] for parent_id in parents]
                configurations = list(itertools.product(*parent_domains))
                cp_matrix = np.empty(shape=(len(self.label_domains[child]), len(configurations)), dtype=np.float32)
                for j, configuration in enumerate(configurations):
                    clf = self.classifiers[child][frozenset(parents)][tuple(configuration)]
                    cp_matrix[:, j] = predict_proba_base_clfs(model=clf, x_test=test_img)
                cpd = TabularCPD(variable=child, variable_card=len(self.label_domains[child]), values=cp_matrix,
                                 evidence=parents,
                                 evidence_card=[len(self.label_domains[parent]) for parent in parents])
                bn.add_cpds(cpd)
            max_prob, assignments = compute_mpe(graph=bn,
                                                variables=['0'],
                                                evidence=None,
                                                elimination_order=None,
                                                show_progress=False)
            for v in assignments.keys():
                pred_labels[i, int(v)] = assignments[v]
            bn.cpds = []
        return pred_labels

    def inference_hamming(self, test_dataloader: IMGDataLoader):
        bn = BayesianNetwork()
        bn.add_nodes_from(self.bn.nodes)
        bn.add_edges_from(self.bn.edges)
        pred_labels = np.empty(shape=(len(test_dataloader.labels), len(self.label_domains)))
        test_img_dataset = test_dataloader.create_img_datasets()
        for i, (test_img, _) in enumerate(tqdm(test_img_dataset)):
            test_img = test_img.unsqueeze(dim=0)
            for child, parents in self.parent_dict.items():
                parent_domains = [self.label_domains[parent_id] for parent_id in parents]
                configurations = list(itertools.product(*parent_domains))
                cp_matrix = np.empty(shape=(len(self.label_domains[child]), len(configurations)), dtype=np.float32)
                for j, configuration in enumerate(configurations):
                    clf = self.classifiers[child][frozenset(parents)][tuple(configuration)]
                    cp_matrix[:, j] = predict_proba_base_clfs(model=clf, x_test=test_img)
                cpd = TabularCPD(variable=child, variable_card=len(self.label_domains[child]), values=cp_matrix,
                                 evidence=parents,
                                 evidence_card=[len(self.label_domains[parent]) for parent in parents])
                bn.add_cpds(cpd)
            inferencer = VariableElimination(bn)
            for k in self.label_domains.keys():
                pred_labels[i, int(k)] = inferencer.map_query(variables=[k], show_progress=False)[k]
            bn.cpds = []
        return pred_labels

    def __create_base_clfs(self, output_dim):
        model = torchvision.models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features, output_dim)
        return model
