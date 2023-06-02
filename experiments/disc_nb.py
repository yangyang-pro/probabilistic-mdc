import os
import sys

sys.path.append('..')

import csv
import pickle
import datetime
import numpy as np
import argparse

from mixed_naive_bayes import MixedNB

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split

from data import TabularDataLoader
from classifier import TabularClassifier
from metrics import hamming_loss_mdc, zero_one_loss_mdc

DATASETS = [
    'Adult',
    'Default',
    'Thyroid',
]

DATASETS_DIR = '../data/MDC'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mixed Data Experiments with Naive Bayes')
    parser.add_argument('dataset', choices=DATASETS, type=str, help='Dataset')
    parser.add_argument('--palim', type=int, default=3, help='The maximum number of parents for each node')
    parser.add_argument('--disclim', type=int, default=3, help='The maximum number of discrete features for each node')
    parser.add_argument('--n-chains', default=10, type=int, help='Number of randomly generated classifier chains')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='Whether or not to plot the BN structure')
    parser.add_argument('--n-folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--output', type=str, default='disc', help='Output path')
    args = parser.parse_args()

    dataset = args.dataset
    base = 'nb'
    palim = args.palim
    disclim = args.disclim
    is_plot = args.plot
    n_chains = args.n_chains
    n_folds = args.n_folds
    output_path = args.output

    base_dir = os.path.join(output_path, base)
    os.makedirs(base_dir, exist_ok=True)
    dataset_dir = os.path.join(base_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    cur_time = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
    model_dir = os.path.join(dataset_dir, 'models', cur_time)
    os.makedirs(model_dir, exist_ok=True)

    csv_file = open(os.path.join(base_dir, dataset + '_' + base + '_' + cur_time + '.csv'), 'w')
    fieldnames = ['Fold',
                  'GBNC-H HL', 'GBNC-H 0/1', 'GBNC-S HL', 'GBNC-S 0/1',
                  'CC HL', 'CC 0/1', 'HL Order', '0/1 Order',
                  'BR HL', 'BR 0/1', 'CP HL', 'CP 0/1',
                  '#Train', '#Test']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    mat_path = os.path.join(DATASETS_DIR, dataset, dataset) + '.mat'
    data_loader = TabularDataLoader(mat_path=mat_path)
    data_loader.labels = data_loader.labels.astype(int)
    max_categories = np.array([len(v) for k, v in data_loader.discrete_feature_domains.items()])

    kf = KFold(n_splits=n_folds, shuffle=True)

    h_hls, h_zos, s_hls, s_zos = [], [], [], []
    cc_hls, cc_zos = [], []
    br_hls, br_zos = [], []
    cp_hls, cp_zos = [], []

    for i, (train_indices, test_indices) in enumerate(kf.split(data_loader.labels)):
        cc_train_indices, cc_val_indices = train_test_split(train_indices, test_size=0.2)
        cc_train_data_loader = data_loader.create_sub_data_loader(cc_train_indices)
        cc_val_data_loader = data_loader.create_sub_data_loader(cc_val_indices)

        # Save train/test indices
        np.savetxt(os.path.join(model_dir, str(i) + '.train.indices'), train_indices, fmt='%i')
        np.savetxt(os.path.join(model_dir, str(i) + '.test.indices'), test_indices, fmt='%i')

        train_data_loader = data_loader.create_sub_data_loader(train_indices)
        test_data_loader = data_loader.create_sub_data_loader(test_indices)
        n_train_samples, n_targets = train_data_loader.labels.shape
        n_test_samples = test_data_loader.labels.shape[0]

        # GBNC
        gbnc = TabularClassifier(palim=palim, disclim=disclim, base=base)
        gbnc.fit_mix(train_data_loader)
        gbnc.learn_structure(plot=is_plot)
        gbnc_h_pred = gbnc.inference_hamming_mix(test_data_loader.continuous_features,
                                                 test_data_loader.discrete_features)
        gbnc_s_pred = gbnc.inference_mix(test_data_loader.continuous_features,
                                         test_data_loader.discrete_features)
        with open(os.path.join(model_dir, str(i) + '.gbnc'), 'wb') as model_file:
            pickle.dump(gbnc, model_file)
        h_hl = hamming_loss_mdc(test_data_loader.labels, gbnc_h_pred)
        h_zo = zero_one_loss_mdc(test_data_loader.labels, gbnc_h_pred)
        s_hl = hamming_loss_mdc(test_data_loader.labels, gbnc_s_pred)
        s_zo = zero_one_loss_mdc(test_data_loader.labels, gbnc_s_pred)
        h_hls.append(h_hl)
        h_zos.append(h_zo)
        s_hls.append(s_hl)
        s_zos.append(s_zo)

        train_disc_features = np.copy(train_data_loader.discrete_features)
        test_disc_features = np.copy(test_data_loader.discrete_features)
        cc_train_disc_features = np.copy(cc_train_data_loader.discrete_features)
        cc_val_disc_features = np.copy(cc_val_data_loader.discrete_features)
        for j in range(train_disc_features.shape[1]):
            encoder = LabelEncoder()
            encoder.fit(data_loader.discrete_features[:, j])
            train_disc_features[:, j] = encoder.transform(train_disc_features[:, j])
            test_disc_features[:, j] = encoder.transform(test_disc_features[:, j])
            cc_train_disc_features[:, j] = encoder.transform(cc_train_disc_features[:, j])
            cc_val_disc_features[:, j] = encoder.transform(cc_val_disc_features[:, j])
        train_features = np.hstack((train_disc_features, train_data_loader.continuous_features))
        test_features = np.hstack((test_disc_features, test_data_loader.continuous_features))

        # BR
        pred_br = []
        brs = []
        for j in range(n_targets):
            br = MixedNB(categorical_features=list(range(train_disc_features.shape[1])), max_categories=max_categories)
            encoder = LabelEncoder()
            train_labels_cur = encoder.fit_transform(train_data_loader.labels[:, j])
            br.fit(train_features, train_labels_cur)
            pred_cur_encoded = br.predict(test_features)
            pred_cur = encoder.inverse_transform(pred_cur_encoded)
            pred_br.append(pred_cur)
            brs.append(br)
        pred_br = np.vstack(pred_br).T
        br_hl = hamming_loss_mdc(y_true=test_data_loader.labels, y_pred=pred_br)
        br_zo = zero_one_loss_mdc(y_true=test_data_loader.labels, y_pred=pred_br)
        print('{:<10}{:<3}{:<10}{:10.3f}{:>10}{:10.3f}'.format('BR', i, 'Hamming', br_hl, 'Subset', br_zo))
        br_hls.append(br_hl)
        br_zos.append(br_zo)
        with open(os.path.join(model_dir, str(i) + '.brs'), 'wb') as model_file:
            pickle.dump(brs, model_file)

        # CP
        encoder = LabelEncoder()
        str_labels = list(map(str, train_data_loader.labels))
        train_labels_encoded = encoder.fit_transform(str_labels)
        cp = MixedNB(categorical_features=list(range(train_disc_features.shape[1])), max_categories=max_categories)
        cp.fit(train_features, train_labels_encoded)
        pred_cp_encoded = cp.predict(test_features)
        pred_cp_str = encoder.inverse_transform(pred_cp_encoded)
        pred_cp = np.array([np.fromstring(label[1:-1], dtype=np.int32, sep=' ') for label in pred_cp_str])
        cp_hl = hamming_loss_mdc(y_true=test_data_loader.labels, y_pred=pred_cp)
        cp_zo = zero_one_loss_mdc(y_true=test_data_loader.labels, y_pred=pred_cp)
        print('{:<10}{:<3}{:<10}{:10.3f}{:>10}{:10.3f}'.format('CP', i, 'Hamming', cp_hl, 'Subset', cp_zo))
        cp_hls.append(cp_hl)
        cp_zos.append(cp_zo)
        with open(os.path.join(model_dir, str(i) + '.cp'), 'wb') as model_file:
            pickle.dump(cp, model_file)

        # CC
        best_hl_order = None
        best_zo_order = None
        best_hl = np.inf
        best_zo = np.inf
        orders = [np.random.permutation(n_targets) for _ in range(n_chains)]
        orders.append(np.arange(n_targets))
        for j, order in enumerate(orders):
            pred_cc = []
            cc_train_disc_features_cur = cc_train_disc_features
            cc_val_disc_features_cur = cc_val_disc_features
            cc_train_features_cur = np.hstack((cc_train_disc_features_cur, cc_train_data_loader.continuous_features))
            cc_val_features_cur = np.hstack((cc_val_disc_features_cur, cc_val_data_loader.continuous_features))
            prev_pred = None
            max_categories_cur = max_categories.copy()
            for k in range(n_targets):
                if k > 0:
                    max_categories_cur = np.append(max_categories_cur,
                                                   len(cc_train_data_loader.label_domains[str(order[k - 1])]))
                    cc_train_disc_features_cur = np.hstack((cc_train_disc_features_cur,
                                                            cc_train_data_loader.labels[:, order[k - 1]][:,
                                                            np.newaxis]))
                    cc_train_features_cur = np.hstack((cc_train_disc_features_cur,
                                                       cc_train_data_loader.continuous_features))
                    cc_val_disc_features_cur = np.hstack((cc_val_disc_features_cur,
                                                          prev_pred[:, np.newaxis]))
                    cc_val_features_cur = np.hstack((cc_val_disc_features_cur, cc_val_data_loader.continuous_features))
                cc = MixedNB(categorical_features=list(range(cc_train_disc_features_cur.shape[1])),
                             max_categories=max_categories_cur)
                encoder = LabelEncoder()
                cc_train_labels_cur = encoder.fit_transform(cc_train_data_loader.labels[:, order[k]])
                cc.fit(cc_train_features_cur, cc_train_labels_cur)
                pred_cur_encoded = cc.predict(cc_val_features_cur)
                pred_cur = encoder.inverse_transform(pred_cur_encoded)
                prev_pred = pred_cur
                pred_cc.append(pred_cur)
            pred_cc = [pred_cc[idx] for idx in order]
            pred_cc = np.vstack(pred_cc).T
            hl = hamming_loss_mdc(y_true=cc_val_data_loader.labels, y_pred=pred_cc)
            zo = zero_one_loss_mdc(y_true=cc_val_data_loader.labels, y_pred=pred_cc)
            print('{:<10}{:<3}{:<10}{:<3}{:<10}{:10.3f}{:>10}{:10.3f}'.format('CC', i, 'Chain', j,
                                                                              'Hamming', hl, 'Subset', zo))
            if hl < best_hl:
                best_hl_order = order
                best_hl = hl
            if zo < best_zo:
                best_zo_order = order
                best_zo = zo
        # Best order in terms of HL
        hl_ccs = []
        prev_pred = None
        pred_hl = []
        train_disc_features_cur = train_disc_features
        test_disc_features_cur = test_disc_features
        train_features_cur = np.hstack((train_disc_features_cur, train_data_loader.continuous_features))
        test_features_cur = np.hstack((test_disc_features_cur, test_data_loader.continuous_features))
        max_categories_cur = max_categories.copy()
        for k in range(n_targets):
            if k > 0:
                max_categories_cur = np.append(max_categories_cur,
                                               len(cc_train_data_loader.label_domains[str(best_hl_order[k - 1])]))
                train_disc_features_cur = np.hstack((train_disc_features_cur,
                                                     train_data_loader.labels[:, best_hl_order[k - 1]][:, np.newaxis]))
                train_features_cur = np.hstack((train_disc_features_cur, train_data_loader.continuous_features))
                test_disc_features_cur = np.hstack((test_disc_features_cur, prev_pred[:, np.newaxis]))
                test_features_cur = np.hstack((test_disc_features_cur, test_data_loader.continuous_features))
            cc = MixedNB(categorical_features=list(range(train_disc_features_cur.shape[1])),
                         max_categories=max_categories_cur)
            encoder = LabelEncoder()
            train_labels_cur = encoder.fit_transform(train_data_loader.labels[:, best_hl_order[k]])
            cc.fit(train_features_cur, train_labels_cur)
            pred_cur_encoded = cc.predict(test_features_cur)
            pred_cur = encoder.inverse_transform(pred_cur_encoded)
            prev_pred = pred_cur
            pred_hl.append(pred_cur)
            hl_ccs.append(cc)
        # Best order in terms of subset 0/1 loss
        zo_ccs = []
        prev_pred = None
        pred_zo = []
        train_disc_features_cur = train_disc_features
        test_disc_features_cur = test_disc_features
        train_features_cur = np.hstack((train_disc_features_cur, train_data_loader.continuous_features))
        test_features_cur = np.hstack((test_disc_features_cur, test_data_loader.continuous_features))
        max_categories_cur = max_categories.copy()
        for k in range(n_targets):
            if k > 0:
                max_categories_cur = np.append(max_categories_cur,
                                               len(cc_train_data_loader.label_domains[str(best_zo_order[k - 1])]))
                train_disc_features_cur = np.hstack((train_disc_features_cur,
                                                     train_data_loader.labels[:, best_zo_order[k - 1]][:, np.newaxis]))
                train_features_cur = np.hstack((train_disc_features_cur, train_data_loader.continuous_features))
                test_disc_features_cur = np.hstack((test_disc_features_cur, prev_pred[:, np.newaxis]))
                test_features_cur = np.hstack((test_disc_features_cur, test_data_loader.continuous_features))
            cc = MixedNB(categorical_features=list(range(train_disc_features_cur.shape[1])),
                         max_categories=max_categories_cur)
            encoder = LabelEncoder()
            train_labels_cur = encoder.fit_transform(train_data_loader.labels[:, best_zo_order[k]])
            cc.fit(train_features_cur, train_labels_cur)
            pred_cur_encoded = cc.predict(test_features_cur)
            pred_cur = encoder.inverse_transform(pred_cur_encoded)
            prev_pred = pred_cur
            pred_zo.append(pred_cur)
            zo_ccs.append(cc)
        pred_hl = [pred_hl[idx] for idx in best_hl_order]
        pred_zo = [pred_zo[idx] for idx in best_zo_order]
        pred_hl = np.vstack(pred_hl).T
        pred_zo = np.vstack(pred_zo).T
        cc_hl = hamming_loss_mdc(test_data_loader.labels, pred_hl)
        cc_zo = zero_one_loss_mdc(test_data_loader.labels, pred_zo)
        cc_hls.append(cc_hl)
        cc_zos.append(cc_zo)

        with open(os.path.join(model_dir, str(i) + '.hl.ccs'), 'wb') as model_file:
            pickle.dump(hl_ccs, model_file)
        with open(os.path.join(model_dir, str(i) + '.zo.ccs'), 'wb') as model_file:
            pickle.dump(zo_ccs, model_file)

        info = {'Fold': i,
                'GBNC-H HL': h_hl, 'GBNC-H 0/1': h_zo, 'GBNC-S HL': s_hl, 'GBNC-S 0/1': s_zo,
                'CC HL': cc_hl, 'CC 0/1': cc_zo, 'HL Order': best_hl_order, '0/1 Order': best_zo_order,
                'BR HL': br_hl, 'BR 0/1': br_zo, 'CP HL': cp_hl, 'CP 0/1': cp_zo,
                '#Train': n_train_samples, '#Test': n_test_samples}
        writer.writerow(info)
        csv_file.flush()
    means = {'Fold': 'Mean',
             'GBNC-H HL': np.mean(h_hls), 'GBNC-H 0/1': np.mean(h_zos),
             'GBNC-S HL': np.mean(s_hls), 'GBNC-S 0/1': np.mean(s_zos),
             'CC HL': np.mean(cc_hls), 'CC 0/1': np.mean(cc_zos),
             'BR HL': np.mean(br_hls), 'BR 0/1': np.mean(br_zos),
             'CP HL': np.mean(cp_hls), 'CP 0/1': np.mean(cp_zos)}
    stds = {'Fold': 'Std',
            'GBNC-H HL': np.std(h_hls), 'GBNC-H 0/1': np.std(h_zos),
            'GBNC-S HL': np.std(s_hls), 'GBNC-S 0/1': np.std(s_zos),
            'CC HL': np.std(cc_hls), 'CC 0/1': np.std(cc_zos),
            'BR HL': np.std(br_hls), 'BR 0/1': np.std(br_zos),
            'CP HL': np.std(cp_hls), 'CP 0/1': np.std(cp_zos)}
    writer.writerow(means)
    writer.writerow(stds)
    csv_file.close()
