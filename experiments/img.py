import os
import sys

sys.path.append('..')

import csv
import datetime
import pickle
import numpy as np
import argparse

from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder

from data import IMGDataLoader
from classifier import IMGClassifier
from utils import predict_img_clfs, train_img_base_clfs
from metrics import hamming_loss_mdc, zero_one_loss_mdc
from temperature_scaling import ModelWithTemperature


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_trainer(model, train_data_loader, n_classes, learning_rate, n_epochs, alpha=1):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=0.1)
    for epoch in tqdm(range(n_epochs), position=0, leave=True):
        model.train()
        n_corrects = 0
        n_samples = 0
        for x_batch, y_batch in train_data_loader:
            optimizer.zero_grad()
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_batch_one_hot = F.one_hot(y_batch.long(), num_classes=n_classes)
            inputs, targets_a, targets_b, lam = mixup_data(x_batch, y_batch_one_hot, alpha=alpha)
            targets_a, targets_b = targets_a.type(torch.FloatTensor), targets_b.type(torch.FloatTensor)
            targets_a, targets_b = targets_a.to(device), targets_b.to(device)
            outputs = model(inputs)
            loss = mixup_criterion(criterion=criterion, pred=outputs, y_a=targets_a, y_b=targets_b, lam=lam)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                _, preds_batch = torch.max(outputs, dim=1)
                n_corrects += torch.sum(preds_batch == y_batch)
                n_samples += len(y_batch)
        tqdm.write('{:<10}{:<5}{:<15}{:<20.3f}'.format('Epoch:', epoch + 1, 'Train ACC:', n_corrects / n_samples),
                   file=sys.stderr)
    model.to(torch.device('cpu'))


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    parser = argparse.ArgumentParser(description='Image Experiments')
    parser.add_argument('dataset', choices=['voc2007'], default='voc2007', type=str, help='Dataset')
    parser.add_argument('--base', choices=['resnet18'], default='resnet18', type=str, help='Base classifier')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--n-epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--palim', type=int, default=3, help='The maximum number of parents for each node')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='Whether or not to plot the BN structure')
    parser.add_argument('--mixup', action=argparse.BooleanOptionalAction, help='Whether or not to use mixup training')
    parser.add_argument('--calibrate', action=argparse.BooleanOptionalAction, help='Whether or not to calibrate')
    parser.add_argument('--output', type=str, default='img', help='Output path')
    args = parser.parse_args()

    dataset = args.dataset
    base = args.base
    batch_size = args.batch_size
    learning_rate = args.lr
    n_epochs = args.n_epochs
    palim = args.palim
    is_plot = args.plot
    is_mixup = args.mixup
    is_calib = args.calibrate
    output_path = args.output

    os.makedirs(output_path, exist_ok=True)
    cur_time = datetime.datetime.now().strftime('%m-%d_%H-%M-%S')
    model_dir = os.path.join(output_path, cur_time)
    os.makedirs(model_dir, exist_ok=True)

    csv_file = open(os.path.join(output_path, cur_time + '.csv'), 'w')
    fieldnames = ['GBNC-H HL', 'GBNC-H 0/1',
                  'GBNC-S HL', 'GBNC-S 0/1',
                  'BR HL', 'BR 0/1',
                  'CP HL', 'CP 0/1']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    TRAIN_IMG_DIR = '../data/VOCdevkit/VOC2007_train/JPEGImages'
    TRAIN_XML_DIR = '../data/VOCdevkit/VOC2007_train/Annotations'
    TEST_IMG_DIR = '../data/VOCdevkit/VOC2007_test/JPEGImages'
    TEST_XML_DIR = '../data/VOCdevkit/VOC2007_test/Annotations'

    TARGET_INFO = {
        'person': ['person'],
        'animal': ['bird', 'cat', 'cow', 'dog', 'horse', 'sheep'],
        'vehicle': ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train'],
        'indoor': ['bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
    }

    train_data_loader = IMGDataLoader(img_dir=TRAIN_IMG_DIR, xml_dir=TRAIN_XML_DIR, target_domains=TARGET_INFO)
    test_data_loader = IMGDataLoader(img_dir=TEST_IMG_DIR, xml_dir=TEST_XML_DIR, target_domains=TARGET_INFO)

    img_classifier = IMGClassifier(palim=len(TARGET_INFO) - 1)
    if is_mixup:
        img_classifier.fit(train_data_loader=train_data_loader,
                           batch_size=batch_size,
                           learning_rate=learning_rate,
                           n_epochs=n_epochs,
                           show_log=True,
                           base_trainer=mixup_trainer,
                           calibration=is_calib)
    else:
        img_classifier.fit(train_data_loader=train_data_loader,
                           batch_size=batch_size,
                           learning_rate=learning_rate,
                           n_epochs=n_epochs,
                           show_log=True,
                           calibration=is_calib)
    img_classifier.learn_structure()

    with open(os.path.join(model_dir, 'gbnc'), 'wb') as gbnc_file:
        pickle.dump(img_classifier, gbnc_file)

    pred_gbnc_s = img_classifier.inference(test_data_loader)
    pred_gbnc_h = img_classifier.inference_hamming(test_data_loader)

    hl_gbnc_h = hamming_loss_mdc(y_true=test_data_loader.labels, y_pred=pred_gbnc_h)
    zo_gbnc_h = zero_one_loss_mdc(y_true=test_data_loader.labels, y_pred=pred_gbnc_h)
    hl_gbnc_s = hamming_loss_mdc(y_true=test_data_loader.labels, y_pred=pred_gbnc_s)
    zo_gbnc_s = zero_one_loss_mdc(y_true=test_data_loader.labels, y_pred=pred_gbnc_s)

    print('{:<20}{:.3f}'.format('GBNC Hamming score:', hl_gbnc_s))
    print('{:<20}{:.3f}'.format('GBNC Exact match score:', zo_gbnc_s))

    br_clfs = []
    for i, target in enumerate(TARGET_INFO.keys()):
        clf = torchvision.models.resnet18(pretrained=True)
        for param in clf.parameters():
            param.requires_grad = False
        clf.fc = torch.nn.Linear(clf.fc.in_features, len(TARGET_INFO[target]) + 1)
        local_data_loader = train_data_loader.create_local_data_loaders(local_files=train_data_loader.img_files,
                                                                        local_labels=train_data_loader.labels[:, i],
                                                                        batch_size=batch_size)
        if is_mixup:
            mixup_trainer(model=clf,
                          train_data_loader=local_data_loader,
                          n_classes=len(TARGET_INFO[target]) + 1,
                          learning_rate=learning_rate,
                          n_epochs=n_epochs)
        else:
            train_img_base_clfs(model=clf,
                                train_data_loader=local_data_loader,
                                learning_rate=learning_rate,
                                n_epochs=n_epochs)
        if is_calib:
            clf = ModelWithTemperature(clf)
            clf.set_temperature(local_data_loader)
        br_clfs.append(clf)
    with open(os.path.join(model_dir, 'br'), 'wb') as br_file:
        pickle.dump(br_clfs, br_file)
    pred_br = []
    for i, target in enumerate(TARGET_INFO.keys()):
        clf = br_clfs[i]
        local_data_loader = test_data_loader.create_local_data_loaders(local_files=test_data_loader.img_files,
                                                                       local_labels=None,
                                                                       batch_size=batch_size,
                                                                       shuffle=False)
        preds_target = predict_img_clfs(model=clf, test_data_loader=local_data_loader)
        pred_br.append(preds_target)
    pred_br = np.vstack(pred_br).T
    hl_br = hamming_loss_mdc(y_true=test_data_loader.labels, y_pred=pred_br)
    zo_br = zero_one_loss_mdc(y_true=test_data_loader.labels, y_pred=pred_br)

    print('{:<20}{:.3f}'.format('BR Hamming loss:', hl_br))
    print('{:<20}{:.3f}'.format('BR 0/1 loss:', zo_br))

    encoder = LabelEncoder()
    str_labels = list(map(str, train_data_loader.labels))
    train_labels_encoded = encoder.fit_transform(str_labels)
    cp_clf = torchvision.models.resnet18(pretrained=True)
    for param in cp_clf.parameters():
        param.requires_grad = False
    cp_clf.fc = torch.nn.Linear(cp_clf.fc.in_features, len(set(train_labels_encoded)))
    cp_train_data_loader = train_data_loader.create_local_data_loaders(local_files=train_data_loader.img_files,
                                                                       local_labels=train_labels_encoded,
                                                                       batch_size=batch_size)
    if is_mixup:
        mixup_trainer(model=cp_clf,
                      train_data_loader=cp_train_data_loader,
                      n_classes=len(set(train_labels_encoded)),
                      learning_rate=learning_rate,
                      n_epochs=n_epochs)
    else:
        train_img_base_clfs(model=cp_clf,
                            train_data_loader=cp_train_data_loader,
                            learning_rate=learning_rate,
                            n_epochs=n_epochs)
    if is_calib:
        cp_clf = ModelWithTemperature(cp_clf)
        cp_clf.set_temperature(cp_train_data_loader)
    cp_test_data_loader = test_data_loader.create_local_data_loaders(local_files=test_data_loader.img_files,
                                                                     local_labels=None,
                                                                     batch_size=batch_size,
                                                                     shuffle=False)
    pred_cp_encoded = predict_img_clfs(model=cp_clf, test_data_loader=cp_test_data_loader)
    pred_cp_str = encoder.inverse_transform(pred_cp_encoded)
    pred_cp = np.array([np.fromstring(label[1:-1], dtype=np.int32, sep=' ') for label in pred_cp_str])
    hl_cp = hamming_loss_mdc(y_true=test_data_loader.labels, y_pred=pred_cp)
    zo_cp = zero_one_loss_mdc(y_true=test_data_loader.labels, y_pred=pred_cp)

    print('{:<20}{:.3f}'.format('CP Hamming loss:', hl_cp))
    print('{:<20}{:.3f}'.format('CP 0/1 loss:', zo_cp))
    with open(os.path.join(model_dir, 'cp'), 'wb') as cp_file:
        pickle.dump(cp_clf, cp_file)

    results = {'GBNC-H HL': hl_gbnc_h,
               'GBNC-H 0/1': zo_gbnc_h,
               'GBNC-S HL': hl_gbnc_s,
               'GBNC-S 0/1': zo_gbnc_s,
               'BR HL': hl_br,
               'BR 0/1': zo_br,
               'CP HL': hl_cp,
               'CP 0/1': zo_cp}
    writer.writerow(results)
    csv_file.flush()
    csv_file.close()
