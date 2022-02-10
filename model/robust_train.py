import torch
from transformers import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import load_dataset, Dataset

import pdb
import numpy as np
import random
import pandas as pd
import os, sys
from collections import Counter

from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from sklearn.metrics import mutual_info_score
import sklearn.metrics as metrics

from model.train import LinearScheduler, masking_fn, stopwords_id, trainer_full, batch_len

from utils.utils import boolean_string, print_args, save_checkpoint, model_eval, load_checkpoint, text_comparison
import utils.logger as logger
import datetime
# from utils.task_eval import 
import argparse

from collections import Counter

import sklearn.metrics as metrics
from sklearn.metrics import f1_score as f1_score_sk
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc
from ood_metrics import fpr_at_95_tpr, calc_metrics


#def onezero_encoder(y, num_classes, ignore_idx=1000):
#    y_onezero = y.new_ones(size=(y.size(0), num_classes)).float()
#    for i in range(y.size(0)):
#        k = y[i]
#        y_onezero[i][k] = 0
#
#    return y_onezero
#
#
#def max_loss(y_onezero, pred):
#    """
#    - x: (batch, seq, seq)
#    - y: (batch, seq, seq)
#    """
#    y_pred = torch.masked_select(pred, y_onezero.bool())
#    y_pred = y_pred / y_onezero.sum(dim=1)
#    loss = -torch.log(y_pred).sum() / pred.size(0)
#
#    return loss

class Purity():

    def count(self, pred):
        self._pred_count = Counter(pred)
        return self._pred_count

    def dist(self, pred_count):
        """probability distribution"""
        pred_len = sum(pred_count.values())
        self._pred_dist = np.array([(i,j/pred_len) for i, j in zip(pred_count.keys(), pred_count.values())])
        return self._pred_dist

    def unique_elements(self, pred, p=1.0):
        #pred = self.sampling_pred(pred, p=p)
        purity = len(set(pred))
        if purity==1:
            detection = True # TP
        else:
            detection = False # TN
        return detection

    def max_purity(self, pred, p=1.0):
        """Counting based"""
        #pred = self.sampling_pred(pred, p=p)
        count = self.count(pred)
        common_element = count.most_common(1)
        return common_element[0][1]/len(pred)

    def entropy(self, pred, p=1.0):
        #pred = self.sampling_pred(pred, p=p)
        count = self.count(pred)
        pred_dist = self.dist(count)
        return stats.entropy(pred_dist[:,1])

    def gini(self, pred, p=1.0):
        #pred = self.sampling_pred(pred, p=p)
        count = self.count(pred)
        pred_dist = self.dist(count)
        out = pred_dist[:,1]**2
        purity = out.sum()
        return purity

#    def mutual_information(self, ):
#        common_element = self.pred_count.most_common(1)
#        common_pred = common_element[0][1]
#        ref = [common_pred for i in range(len(purity.pred))]
#
#        return mutual_info_score(ref, self.pred)

def top_k_entropy(pred_list, topk):

    purity_ = Purity()
    entropy_list = []
    for pred in pred_list:
        ent_val = purity_.entropy(pred[:topk])
        entropy_list.append(ent_val)
    return np.array(entropy_list)


def adv_dataset_stat(df):
    df_adv = df[df['result_type']=='Successful'].reset_index()
    count, length = text_comparison(df_adv['orig'], df_adv['pert'])
    pert_ratio = count/length
    avg_query = df_adv['n_query'].mean()
    print(f"AvgPertRatio: {pert_ratio.mean()*100:.4f} || AvgQuery: {int(avg_query)}") 

class PurityStat():
    def __init__(self, ):
        self.reset()

    def reset(self,):
        self.tp = []
        self.tn = []
        self.pred = []

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

    def eval(self, t=0.8, method='entropy'):
        TP = np.array(self.tp)
        TN = np.array(self.tn)

        TPR = []
        FPR = []
        if method=='entropy':
            TP_ = TP<t
            TPR.append(TP_.sum()/TP.shape[0])
            FP_ = TN<t
            FPR.append(FP_.sum()/TN.shape[0])
        elif method=='hard_purity':
            TP_ = TP==True
            TPR.append(TP_.sum()/TP.shape[0])
            FP_ = TN==True
            FPR.append(FP_.sum()/TN.shape[0])
        else:
            TP_ = TP>t
            TPR.append(TP_.sum()/TP.shape[0])
            FP_ = TN>t
            FPR.append(FP_.sum()/TN.shape[0])

        TPR = np.array(TPR)
        FPR = np.array(FPR)

        self.tp_sum = TP_.sum()
        self.fp_sum = FP_.sum()

        self.tp_count = TP.shape[0]
        self.tn_count = TN.shape[0]

        self.tpr = TPR
        self.fnr = 1-TPR
        self.fpr = FPR
        self.tnr = 1-FPR

        n_tp = self.tp_sum
        n_tn = self.tn_count - self.fp_sum

        log = f"Method: {method} || Threshold: {t} || TP: {n_tp} || TN: {n_tn}"
        print(log)

        return n_tp, n_tn

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

    def eval(self, direction='down', t=0.8):
        TP = np.array(self.tp)
        TN = np.array(self.tn)

        TPR = []
        FPR = []

        if direction=='down':
            TP_ = TP<t
            TPR.append(TP_.sum()/TP.shape[0])
            FP_ = TN<t
            FPR.append(FP_.sum()/TN.shape[0])
        else:
            TP_ = TP>t
            TPR.append(TP_.sum()/TP.shape[0])
            FP_ = TN>t
            FPR.append(FP_.sum()/TN.shape[0])

        TPR = np.array(TPR)
        FPR = np.array(FPR)

        self.tp_sum = TP_.sum()
        self.fp_sum = FP_.sum()

        self.tp_count = TP.shape[0]
        self.tn_count = TN.shape[0]

        self.tpr = TPR
        self.fnr = 1-TPR
        self.fpr = FPR
        self.tnr = 1-FPR

        n_tp = self.tp_sum
        n_tn = self.tn_count - self.fp_sum

        log = f"Threshold: {t} || TP: {n_tp} || TN: {n_tn}"
        print(log)
        return n_tp, n_tn

    def print_perf(self, args):

        pert = np.array(self.tp)
        clean = np.array(self.tn)

#        test = np.vstack([pert, clean])
        test = np.array(self.pred)
        feature = test[:,0]
        y_true = test[:,1]

        if args.conf_feature=='org':
            print("MSP is used")
            feature = 1-feature

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

        dir_path = "./data/features/"
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        f_name = f"pert_{args.dataset}_{args.attack_method}_{args.load_model}_{args.det_mode}.npy"
        path = os.path.join(dir_path, f_name)
        with open(path, 'wb') as f:
            np.save(f, pert)

        f_name = f"clean_{args.dataset}_{args.attack_method}_{args.load_model}_{args.det_mode}.npy"
        path = os.path.join(dir_path, f_name)
        with open(path, 'wb') as f:
            np.save(f, clean)



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

