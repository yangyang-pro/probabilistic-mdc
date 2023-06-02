import os
import sys

sys.path.append('..')

import csv
import pickle
import datetime
import numpy as np
import argparse

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, train_test_split
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset

from data import TabularDataLoader
from classifier import TabularClassifier
from metrics import hamming_loss_mdc, zero_one_loss_mdc

DATASETS = [
    'Edm',
    'Jura',
    'Enb',
    'Voice',
    'Song',
    'Flickr',
    'Fera',
    'WQplants',
    'WQanimals',
    'Rf1',
    'Pain',
    'Disfa',
    'WaterQuality',
    'Oes97',
    'Oes10',
    'Scm20d',
    'Scm1d',
]

DATASETS_DIR = '../data/MDC'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tabular Continuous Features Experiments')
    parser.add_argument('dataset', choices=DATASETS, type=str, help='Dataset')
    parser.add_argument('--base', choices=['lr', 'nb'], default='nb', type=str, help='Base classifier')
    parser.add_argument('--palim', type=int, default=3, help='The maximum number of parents for each node')
    parser.add_argument('--n-chains', default=10, type=int, help='Number of randomly generated classifier chains')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='Whether or not to plot the BN structure')
    parser.add_argument('--n-folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--output', type=str, default='cont', help='Output path')
    args = parser.parse_args()

    dataset = args.dataset
    base = args.base
    palim = args.palim
    is_plot = args.plot
    n_chains = args.n_chains
    n_folds = args.n_folds
    output_path = args.output

    base_clf = SGDClassifier(loss='log_loss', n_jobs=-1) if base == 'lr' else GaussianNB()

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
        gbnc = TabularClassifier(palim=palim, base=base)
        gbnc.fit(train_data_loader)
        gbnc.learn_structure(plot=is_plot)
        gbnc_h_pred = gbnc.inference_hamming(test_data_loader.continuous_features)
        gbnc_s_pred = gbnc.inference(test_data_loader.continuous_features)
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

        # BR
        br = BinaryRelevance(classifier=base_clf)
        br.fit(X=train_data_loader.continuous_features, y=train_data_loader.labels)
        pred = br.predict(X=test_data_loader.continuous_features)
        br_hl = hamming_loss_mdc(y_true=test_data_loader.labels, y_pred=pred)
        br_zo = zero_one_loss_mdc(y_true=test_data_loader.labels, y_pred=pred)
        print('{:<10}{:<3}{:<10}{:10.3f}{:>10}{:10.3f}'.format('BR', i, 'Hamming', br_hl, 'Subset', br_zo))
        br_hls.append(br_hl)
        br_zos.append(br_zo)
        with open(os.path.join(model_dir, str(i) + '.br'), 'wb') as model_file:
            pickle.dump(br, model_file)

        # CP
        cp = LabelPowerset(classifier=base_clf)
        cp.fit(X=train_data_loader.continuous_features, y=train_data_loader.labels)
        pred = cp.predict(X=test_data_loader.continuous_features)
        cp_hl = hamming_loss_mdc(y_true=test_data_loader.labels, y_pred=pred)
        cp_zo = zero_one_loss_mdc(y_true=test_data_loader.labels, y_pred=pred)
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
        for j in range(n_chains):
            order = np.random.permutation(n_targets)
            cc = ClassifierChain(classifier=base_clf, order=order)
            cc.fit(X=cc_train_data_loader.continuous_features, y=cc_train_data_loader.labels)
            pred = cc.predict(X=cc_val_data_loader.continuous_features)
            hl = hamming_loss_mdc(y_true=cc_val_data_loader.labels, y_pred=pred)
            zo = zero_one_loss_mdc(y_true=cc_val_data_loader.labels, y_pred=pred)
            print('{:<10}{:<3}{:<10}{:<3}{:<10}{:10.3f}{:>10}{:10.3f}'.format('CC', i, 'Chain', j,
                                                                              'Hamming', hl, 'Subset', zo))
            if hl < best_hl:
                best_hl_order = order
                best_hl = hl
            if zo < best_zo:
                best_zo_order = order
                best_zo = zo
        cc = ClassifierChain(classifier=base_clf)
        cc.fit(X=cc_train_data_loader.continuous_features, y=cc_train_data_loader.labels)
        pred = cc.predict(X=cc_val_data_loader.continuous_features)
        hl = hamming_loss_mdc(y_true=cc_val_data_loader.labels, y_pred=pred)
        zo = zero_one_loss_mdc(y_true=cc_val_data_loader.labels, y_pred=pred)
        print('{:<10}{:<3}{:<10}{:<3}{:<10}{:10.3f}{:>10}{:10.3f}'.format('CC', i, 'Chain', 'N',
                                                                          'Hamming', hl, 'Subset', zo))
        if hl < best_hl:
            best_hl_order = np.arange(n_targets)
            best_hl = hl
        if zo < best_zo:
            best_zo_order = np.arange(n_targets)
            best_zo = zo
        hl_cc = ClassifierChain(classifier=base_clf, order=best_hl_order)
        zo_cc = ClassifierChain(classifier=base_clf, order=best_zo_order)
        hl_cc.fit(train_data_loader.continuous_features, train_data_loader.labels)
        zo_cc.fit(train_data_loader.continuous_features, train_data_loader.labels)
        pred_hl = hl_cc.predict(test_data_loader.continuous_features)
        pred_zo = zo_cc.predict(test_data_loader.continuous_features)
        cc_hl = hamming_loss_mdc(test_data_loader.labels, pred_hl)
        cc_zo = zero_one_loss_mdc(test_data_loader.labels, pred_zo)
        cc_hls.append(cc_hl)
        cc_zos.append(cc_zo)

        with open(os.path.join(model_dir, str(i) + '.hl.cc'), 'wb') as model_file:
            pickle.dump(hl_cc, model_file)
        with open(os.path.join(model_dir, str(i) + '.zo.cc'), 'wb') as model_file:
            pickle.dump(zo_cc, model_file)

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
