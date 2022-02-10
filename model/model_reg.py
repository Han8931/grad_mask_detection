import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW
#from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertForMaskedLM
from transformers import DistilBertForMaskedLM

from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _WeightedLoss

import pdb
import numpy as np
import random
import pandas as pd
import os, sys

from model.train import LinearScheduler, masking_fn, batch_len

import math

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ClsText(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(ClsText, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device
        self.pad_idx = args.pad_idx
        self.mask_idx = args.mask_idx
        #self.batchnorm = nn.BatchNorm1d(768, affine=False)

    def embeddings(self, input_ids, attention_mask):
        outputs = self.enc.encoder(input_ids, attention_mask, output_hidden_states=True)
        embs = outputs['hidden_states']
        emb_init = embs[0]
        emb_last = embs[-1]

        return emb_last, emb_init

    def forward(self, input_ids, attention_mask, labels=None):
        """
        - x: input
        """
        emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)
        logits = self.cls(emb_x) # output from a FC layer
        
        if labels is not None:

            #loss_fn = nn.MSELoss()
            loss_fn = nn.CrossEntropyLoss()
            loss_ce = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

#            loss_fn = SmoothCrossEntropyLoss(smoothing=0.1)
#            loss_ce = loss_fn(logits.view(-1, self.num_classes).softmax(dim=-1), labels.view(-1))

            output = {'logits': logits, 'loss': loss_ce}

            return output

        output = {'logits': logits}

        return output


class Encoder(nn.Module):
    def __init__(self, encoder):
        super(Encoder, self).__init__()
        self.encoder = encoder

#        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
#        pooled_output = hidden_state[:, 0]  # (bs, dim)

    def forward(self, input_ids):
        enc_out = self.encoder(input_ids)
        hidden_state = enc_out[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        return pooled_output

class ClsRegText2(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(ClsRegText2, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device
        self.pad_idx = args.pad_idx

    def embeddings(self, input_ids, attention_mask):
        outputs = self.enc.encoder(input_ids, attention_mask, output_hidden_states=True)
        embs = outputs['hidden_states']
        emb_init = embs[0]
        emb_last = embs[-1]

        return emb_last, emb_init

    def onehot_encoder(self, y):
        y_onehot = y.new_zeros(size=(y.size(0), self.num_classes)).float()
        for i in range(y.size(0)):
            k = y[i].item()
            Q = k//self.num_classes
            R = k%self.num_classes
            if Q==0:
                #y_onehot[i][k] = 4.0#+torch.randn_like(k.float())*0.2
                y_onehot[i][k] = 1.0#Best
            else:
                y_onehot[i][R] = 1.0+Q*0.5#+torch.randn_like(k.float())*0.4
                #y_onehot[i][R] = 2.0+Q*1.5#Best

        return y_onehot

    def forward(self, input_ids, attention_mask, labels=None):
        """
        - x: input
        """
        emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)
        logits = self.cls(emb_x) # output from a FC layer

        if labels is not None:
            labels = self.onehot_encoder(labels)
            #labels = label_oh+torch.randn_like(label_oh)*1.0
            #labels = label_oh+torch.randn_like(label_oh)*0.3

            loss_fn = nn.MSELoss()
            loss = loss_fn(logits.squeeze(1), labels)

            output = {'logits': logits, 'loss': loss, 'emb': emb_x}

            return output

        output = {'logits': logits, 'emb': emb_x}

        return output

class ClsRegText(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(ClsRegText, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device
        self.pad_idx = args.pad_idx

    def onehot_encoder(self, labels, w_l=1):
#        y_onehot = labels.new_zeros(size=(labels.size(0), self.num_classes)).float()
#        y_onehot[range(labels.size(0)), labels] = (labels.new_ones(size=(labels.size(0),))*w_l).float()
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        return y_onehot

    def forward(self, input_ids, attention_mask, labels=None):
        """
        - x: input
        """
        emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)
        logits = self.cls(emb_x) # output from a FC layer

        if labels is not None:

            loss_fn = nn.MSELoss()

            y_onehot = self.onehot_encoder(labels)
            #y_onehot[range(labels.size(0)), labels] = (labels*3).float()

            loss = loss_fn(logits, y_onehot)
            #loss = loss_fn(logits, labels.float().unsqueeze(1))

            output = {'logits': logits, 'loss': loss}

            return output

        output = {'logits': logits}

        return output

class RAT(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(RAT, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device
        self.pad_idx = args.pad_idx
        self.label_noise = args.label_noise
        #self.batchnorm = nn.BatchNorm1d(768, affine=False)

    def embeddings(self, input_ids, attention_mask):
        outputs = self.enc.encoder(input_ids, attention_mask, output_hidden_states=True)
        embs = outputs['hidden_states']
        emb_init = embs[0]
        emb_last = embs[-1]

        return emb_last, emb_init

    def onehot_encoder(self, y):
        y_onehot = y.new_zeros(size=(y.size(0), self.num_classes)).float()
        for i in range(y.size(0)):
            k = y[i].item()
            Q = k//self.num_classes
            R = k%self.num_classes
            if Q==0:
                #y_onehot[i][k] = 4.0#+torch.randn_like(k.float())*0.2
                y_onehot[i][k] = 4.0#Best
                #y_onehot[i][k] = 3.0#Best
            else:
                #y_onehot[i][R] = 2.0+Q*torch.randn(1).item()#*1.5#+torch.randn_like(k.float())*0.4
                #y_onehot[i][R] = 6.0

                #y_onehot[i][R] = 2.0+Q*4.0#Best Q*torch.randn(1).item()
                y_onehot[i][R] = 4.0+Q*0.5#+Q*torch.randn(1).item()
        return y_onehot

    def forward(self, input_ids, attention_mask, labels=None):
        """
        - x: input
        """
        emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)
        #emb_x = self.batchnorm(emb_x)
        logits = self.cls(emb_x) # output from a FC layer

        if labels is not None:
            #labels = self.onehot_encoder(labels)
            #labels = labels.float()
            #labels = labels+torch.randn_like(labels)*0.2

            labels = self.onehot_encoder(labels)

            #label_oh = self.onehot_encoder(labels)
            #labels = label_oh+torch.randn_like(label_oh)*0.5
            #labels = label_oh+torch.randn_like(label_oh)*0.3

            loss_fn = nn.MSELoss()
            loss = loss_fn(logits.squeeze(1), labels)

            output = {'logits': logits, 'loss': loss}

            return output

        output = {'logits': logits}

        return output

class SeqRegModel(nn.Module):
    """
    - Simple BERT based text classification model.
    """
    def __init__(self, input_size: int, output_size: int, dropout=0.1):
        super(SeqRegModel, self).__init__()

        #self.fc_1 = nn.Linear(input_size, input_size)
        self.fc_2 = nn.Linear(input_size, output_size)
        #self.fc_2 = nn.Linear(input_size, output_size)
        #self.sigmoid = nn.Sigmoid()

#        if dropout != 0:
#            self.dropout_p = dropout
#            self.dropout = nn.Dropout(dropout)
#        else:
#            self.dropout = False

        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):

#        out = self.fc_1(x)
#        out = nn.ReLU()(out)

#        if self.dropout != False:
#            out = self.dropout(x)

        out = self.fc_2(x) # Output logits
        #out = self.fc_2(x) # Output logits

        return out

class SeqClsModel2(nn.Module):
    """
    - Simple BERT based text classification model.
    """
    def __init__(self, input_size: int, output_size: int, dropout=0.1):
        super(SeqClsModel2, self).__init__()

        self.fc_2 = nn.Linear(input_size, output_size)
        #self.sigmoid = nn.Sigmoid()

#        if dropout != 0:
#            self.dropout_p = dropout
#            self.dropout = nn.Dropout(dropout)
#        else:
#            self.dropout = False

        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):

#        out = self.fc_1(x)
#        out = nn.ReLU()(out)
#
#        if self.dropout != False:
#            out = self.dropout(out)

        out = self.fc_2(x) # Output logits

        return out

class SeqClsModel(nn.Module):
    """
    - Simple BERT based text classification model.
    """
    def __init__(self, input_size: int, output_size: int, dropout=0.1):
        super(SeqClsModel, self).__init__()

        self.fc_1 = nn.Linear(input_size, input_size)
        self.fc_2 = nn.Linear(input_size, output_size)
        #self.sigmoid = nn.Sigmoid()

        if dropout != 0:
            self.dropout_p = dropout
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = False

        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):

        out = self.fc_1(x)
        out = nn.ReLU()(out)

        if self.dropout != False:
            out = self.dropout(out)

        out = self.fc_2(out) # Output logits

        return out

