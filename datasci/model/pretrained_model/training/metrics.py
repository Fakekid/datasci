# coding:utf8

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def cls_metrics(labels, logits, mask=None, data_on_gpu=True, return_dict=True):
    if data_on_gpu:
        labels = labels.cpu().numpy()
        logits = logits.detach().cpu().numpy()
    else:
        labels = labels.numpy()
        logits = logits.detach().numpy()

    all_labels = [x for x in range(logits.shape[-1])]
    preds = np.argmax(logits, axis=-1)
    labels_onehot = label_binarize(labels, all_labels)

    acc = np.mean(preds == labels)

    p = precision_score(y_true=labels, y_pred=preds, average='macro', zero_division=0)
    r = recall_score(y_true=labels, y_pred=preds, average='macro', zero_division=0)

    try:
        auc = roc_auc_score(y_true=labels_onehot, y_score=logits, average='macro')
    except ValueError as e:
        auc = -1
        # print(e)

    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')

    if return_dict:
        return {'acc': acc, 'p': p, 'r': r, 'auc': auc, 'f1': f1}
    else:
        return acc, p, r, auc, f1


def cls_metrics_seq(labels, logits, mask=None, data_on_gpu=True, return_dict=True):
    if data_on_gpu:
        labels = labels.cpu().numpy()
        logits = logits.detach().cpu().numpy()
    else:
        labels = labels.numpy()
        logits = logits.detach().numpy()

    all_labels = [x for x in range(logits.shape[-1])]
    preds = np.argmax(logits, axis=-1)

    if mask is not None:
        acc = np.sum((preds == labels) * mask.numpy()) / np.sum(mask.numpy())
    else:
        acc = np.mean(preds == labels)

    seq_p, seq_r, seq_auc, seq_f1 = 0, 0, 0, 0
    for idx, seq_logits in enumerate(logits):
        seq_preds = preds[idx]
        seq_labels = labels[idx]

        seq_p += precision_score(seq_labels, seq_preds, labels=all_labels[1:], average='macro', zero_division=0)
        seq_r += recall_score(seq_labels, seq_preds, labels=all_labels[1:], average='macro', zero_division=0)

        seq_labels_onehot = label_binarize(seq_labels, all_labels)
        try:
            seq_auc += roc_auc_score(y_true=seq_labels_onehot, y_score=seq_logits, average='macro')
        except ValueError as e:
            seq_auc += 0
        seq_f1 += f1_score(seq_labels, seq_preds, average='macro')

    p, r, auc, f1 = \
        seq_p / logits.shape[0], seq_r / logits.shape[0], seq_auc / logits.shape[0], seq_f1 / logits.shape[0]

    if return_dict:
        return {'acc': acc, 'p': p, 'r': r, 'auc': auc, 'f1': f1}
    else:
        return acc, p, r, auc, f1
