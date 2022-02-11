import pdb
import numpy as np
import random
import pandas as pd
import os, sys

import sklearn.metrics as metrics
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc
from ood_metrics import fpr_at_95_tpr, calc_metrics

class DetectionStat():
    def __init__(self, ):
        self.reset()

    def reset(self,):
        self.tp = []
        self.tn = []
        self.pred  = []

    def update2(self, val, label):
        for v, l in zip(val, label):
            if l.item()==0:
                self.tn.append(v)
            elif l.item()==1:
                self.tp.append(v)
            else:
                print("Label Error 1/0")

            self.pred.append((v, l.item()))

    def update(self, val, label):
        if label.item()==0:
            self.tn.append(val)
        elif label.item()==1:
            self.tp.append(val)
        else:
            print("Label Error 1/0")

        self.pred.append((val, label.item()))

    def stat(self,):
        TP = np.array(self.tp)
        TN = np.array(self.tn)

        self.tp_avg = TP.mean()
        self.tn_avg = TN.mean()
        self.tp_std = TP.std()
        self.tn_std = TN.std()

    def print_perf(self, args):

        pert = np.array(self.tp)
        clean = np.array(self.tn)

#        test = np.vstack([pert, clean])
        test = np.array(self.pred)
        feature = test[:,0]
        y_true = test[:,1]

        auroc = metrics.roc_auc_score(y_true, feature, average='macro')

        fpr, tpr, thresholds = metrics.roc_curve(y_true, feature, pos_label=1)
        fpr95 = fpr_at_95_tpr(feature, y_true)
        print(f"AUROC: {auroc*100:.4f} || FPR95: {fpr95*100:.4f}")

        output = calc_metrics(feature, y_true)
        AUPR_In = output['aupr_in']
        AUPR_Out = output['aupr_out']
        print(f"AURP_In: {AUPR_In*100:.4f} || AUPR_Out: {AUPR_Out*100:.4f}")

        error = detection_error(feature, y_true)
        acc = 1-error
        eer = compute_eer(y_true, feature, positive_label=1)
        print(f"Acc: {acc*100:.2f}")
        print(f"EER: {eer*100:.2f}")

        print(f"Pert: {pert.mean()*100:.2f}/{pert.std()*100:.2f} || Clean: {clean.mean()*100:.2f}/{clean.std()*100:.2f}", flush=True)
        print(f"AUROC: {auroc*100:.2f} || EER: {eer*100:.2f} || FPR95: {fpr95*100:.2f}", flush=True)


def detection_error(preds, labels):
    """Return the misclassification probability when TPR is 95%.

    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
            Negatives are assumed to be labelled as 1
    """
    fpr, tpr, _ = metrics.roc_curve(labels, preds)

    # Get ratios of positives to negatives
    neg_ratio = sum(np.array(labels) == 1) / len(labels)
    pos_ratio = 1 - neg_ratio

    # Get indexes of all TPR >= 95%
    idxs = [i for i, x in enumerate(tpr) if x>=0.50]

    # Calc error for a given threshold (i.e. idx)
    # Calc is the (# of negatives * FNR) + (# of positives * FPR)
    _detection_error = lambda idx: neg_ratio * (1 - tpr[idx]) + pos_ratio * fpr[idx]

    # Return the minimum detection error such that TPR >= 0.95
    return min(map(_detection_error, idxs))

def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = metrics.roc_curve(label, pred, positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2

    return eer

