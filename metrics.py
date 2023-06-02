import numpy as np

from sklearn.metrics import confusion_matrix


def hamming_score_mdc(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.size


def hamming_loss_mdc(y_true, y_pred):
    return 1 - (np.sum(y_true == y_pred) / y_true.size)


def exact_match_mdc(y_true, y_pred):
    return np.sum(np.all(y_true == y_pred, axis=1)) / len(y_true)


def zero_one_loss_mdc(y_true, y_pred):
    return 1 - (np.sum(np.all(y_true == y_pred, axis=1)) / len(y_true))


def f1_score_mdc(y_true, y_pred, average):
    """
    TODO: This method needs to be double-checked.
    :param y_true:
    :param y_pred:
    :param average:
    :return:
    """
    n_targets = y_true.shape[1]
    tps = np.zeros(n_targets)
    fps = np.zeros(n_targets)
    fns = np.zeros(n_targets)
    tns = np.zeros(n_targets)
    for j in range(n_targets):
        cm = confusion_matrix(y_true=y_true[:, j], y_pred=y_pred[:, j])
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tn = cm.sum() - (tp + fp + fn)
        tps[j] = tp.sum()
        fps[j] = fp.sum()
        fns[j] = fn.sum()
        tns[j] = tn.sum()
    if average == 'macro':
        precisions = tps / (tps + fps)
        recalls = tps / (tps + fns)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        return (2 * mean_precision * mean_recall) / (mean_precision + mean_recall)
    elif average == 'micro':
        global_precision = tps.sum() / np.sum(tps + fps)
        global_recall = tps.sum() / np.sum(tps + fns)
        return (2 * global_precision * global_recall) / (global_precision + global_recall)
    else:
        raise ValueError('Wrong averaging method.')


def sub_exact_match_mdc(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    n_samples, n_targets = y_pred.shape
    corrects = np.sum(y_true == y_pred, axis=1)
    return np.sum(corrects >= n_targets - 1) / n_samples
