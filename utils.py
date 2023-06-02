import sys
import itertools
import numpy as np

import torch

from tqdm import tqdm

from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss

from torch import softmax, log_softmax


def powerset(iterable, lim=3):
    s = list(iterable)
    return [tuple(sorted(i)) for i in itertools.chain.from_iterable(itertools.combinations(s, r)
                                                                    for r in range(lim + 1))]


def subset(iterable, n_elements):
    s = list(iterable)
    return [tuple(sorted(i)) for i in itertools.chain.from_iterable(itertools.combinations(s, r)
                                                                    for r in range(n_elements, n_elements + 1))]


def gen_pairwise_combinations(candidates):
    pairwise_combinations = []
    for i in range(len(candidates) - 1):
        for j in range(i + 1, len(candidates)):
            candidate_prev, candidate_cur = candidates[i], candidates[j]
            if len(set(candidate_prev).intersection(set(candidate_cur))) == len(candidate_prev) - 1:
                pairwise_combinations.append(tuple(sorted(set(candidate_prev + candidate_cur))))
    return list(set(pairwise_combinations))


def train_tabular_base_clfs(model, x_train, y_train, learning_rate=None, n_epochs=None):
    if isinstance(model, torch.nn.Module):
        if learning_rate is None or n_epochs is None:
            raise ValueError('Not specified training hyper-parameters')
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=0.1)
        for epoch in range(1, n_epochs + 1):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            cll_train = -loss.item()
            loss.backward()
            optimizer.step()
            # if epoch % 100 == 0:
            #     model.eval()
            #     with torch.no_grad():
            #         outputs = model(X_train)
            #         _, preds = torch.max(outputs, dim=1)
            #         acc_train = torch.sum(preds == Y_train) / len(Y_train)
            #     print('Epoch', epoch, 'Training CLL', cll_train, 'Training ACC', acc_train)
    elif isinstance(model, ClassifierMixin):
        model.fit(x_train, y_train)
    elif isinstance(model, list):
        return
    else:
        raise NotImplementedError


def train_img_base_clfs(model, train_data_loader, learning_rate, n_epochs):
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
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
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


def predict_img_clfs(model, test_data_loader):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for x_batch in test_data_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            _, preds_batch = torch.max(outputs, dim=1)
            preds.append(preds_batch)
    preds = torch.hstack(preds)
    model.to(torch.device('cpu'))
    return preds.detach().cpu().numpy()


def compute_cll_tabular_clfs(model, x_train, y_train):
    if isinstance(model, torch.nn.Module):
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        model.eval()
        with torch.no_grad():
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            return -loss.item()
    elif isinstance(model, ClassifierMixin):
        pred_probs = model.predict_proba(x_train)
        return -log_loss(y_train, pred_probs, normalize=False)
    elif isinstance(model, list):
        pred_probs = np.zeros((len(y_train), len(model)))
        pred_probs[np.arange(len(y_train))] = model
        return -log_loss(y_train, pred_probs, normalize=False)
    else:
        raise NotImplementedError


def compute_cll_img_clfs(model, train_data_loader):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    cll = 0.0
    with torch.no_grad():
        for x_batch, y_batch in train_data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            cll += -loss.item()
    model.to(torch.device('cpu'))
    return cll


def predict_proba_base_clfs(model, x_test):
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            probs = softmax(model(x_test), dim=1)
            return probs.detach().numpy()
    elif isinstance(model, ClassifierMixin):
        x = x_test.reshape(1, -1) if len(x_test.shape) == 1 else x_test
        return model.predict_proba(x)
    elif isinstance(model, list):
        return np.array(model)
    else:
        raise NotImplementedError


def predict_log_proba_base_clfs(model, x_test):
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            probs = log_softmax(model(x_test), dim=1)
            return probs.detach().numpy()
    elif isinstance(model, ClassifierMixin):
        x = x_test.reshape(1, -1) if len(x_test.shape) == 1 else x_test
        return model.predict_log_proba(x)
    elif isinstance(model, list):
        return np.log(model)
    else:
        raise NotImplementedError
