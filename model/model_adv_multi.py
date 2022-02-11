import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW
#from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertForMaskedLM
from transformers import DistilBertForMaskedLM
from transformers import RobertaTokenizer, RobertaModel

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
#from transformers.modeling_roberta import 

from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification

from torch.utils.data import DataLoader
from torch.nn.modules.loss import _WeightedLoss

import pdb
import numpy as np
import random
import pandas as pd
import os, sys

from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

from model.train import LinearScheduler, batch_len
#from model.robust_train import onezero_encoder, max_loss

import math

from nltk.tokenize import word_tokenize

from abc import ABC, abstractmethod

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

    def forward(self, input_ids, attention_mask, labels=None):
        """
        - x: input
        """
        emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)
        #emb_bn_x = self.batchnorm(emb_x)
        logits = self.cls(emb_x) # output from a FC layer
        

        if labels is not None:

            # Cross Entropy
            #loss_fn = nn.MSELoss()
            loss_fn = nn.CrossEntropyLoss()
            loss_ce = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

#            loss_fn = SmoothCrossEntropyLoss(smoothing=0.1)
#            loss_ce = loss_fn(logits.view(-1, self.num_classes).softmax(dim=-1), labels.view(-1))

            output = {'logits': logits, 'loss': loss_ce, 'emb': emb_x}

            return output

        output = {'logits': logits, 'emb': emb_x}

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

class ClsTextDet(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(ClsTextDet, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device
        self.pad_idx = args.pad_idx
        self.mask_idx = args.mask_idx

    def grad_detection_batch2(self, input_ids, attention_mask, topk=1.0, pred=None, mask_filter=False):
        """
        - x: input
        """
        b_length = batch_len(input_ids, self.pad_idx)

        if topk<1.0 or topk==1.0:
            topk = int(b_length[0].item()*topk)
        else:
            topk = int(topk)

        for b_len in b_length:
            if b_len.item()<topk:
                topk = b_len.item()-1

        if pred is None:
            with torch.no_grad():
                out_dict = self.forward(input_ids, attention_mask)
            logits = out_dict['logits']
            pred = logits.softmax(dim=-1).argmax(dim=-1) # prediction label

        else:
            if input_ids.shape[0]==1:
                pred = torch.Tensor([pred]).long().to(self.device)

        delta_grad_ = self.get_emb_grad(input_ids, attention_mask, pred)
        delta_grad = delta_grad_[0].detach()
        norm_grad = torch.norm(delta_grad, p=2, dim=-1)

        indice_list = []
        for i, len_ in enumerate(b_length):
            if len_>20:
                val, indices_ = torch.topk(norm_grad[i, :len_], 10)
            else:
                val, indices_ = torch.topk(norm_grad[i, :len_], topk)

            last_token = len_.item()-1 # [SEP]
            if mask_filter:
                ind = [x.item() for x in indices_ if x <= len_.item() and x != last_token and x != 0]
            else:
                ind = [x.item() for x in indices_]

            indice_list.append(ind)
            
        return indice_list, norm_grad

    def get_emb_grad(self, input_ids, attention_mask, labels):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """

        self.enc.eval()

        embedding_layer = self.enc.encoder.get_input_embeddings()
        #embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad

        embedding_layer.weight.requires_grad = True

        emb_grads = []
        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        #emb_hook = embedding_layer.register_backward_hook(grad_hook)
        emb_hook = embedding_layer.register_full_backward_hook(grad_hook)
        self.enc.zero_grad()

        output = self.forward(input_ids, attention_mask, labels=labels)
        loss = output['loss']

        loss.backward()

        # grad w.r.t to word embeddings
        #grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove() # Remove Hook

        return emb_grads

    def iterative_grad_mask_detection_batch(self, input_ids, attention_mask, pred_org, topk=10, indices=None, multi_mask=0):
        """
        - x: input
        """
        b_length = batch_len(input_ids, self.pad_idx)

        masked_ids = input_ids.clone()
        for i, _ in enumerate(masked_ids):
            masked_ids[i][indices[i][0]] = self.mask_idx

        prediction_list = []
        confidence_list = []

        for m_idx in range(multi_mask):

            output = self.forward(masked_ids, attention_mask)
            logits = output['logits']

            conf_l = [] # each input_ids
            pred_l = []
            pred = logits.argmax(dim=-1) # output from a FC layer
            smp = logits.softmax(dim=-1)

            for i, p in enumerate(pred):
                conf = smp[i, pred_org[i].item()]
                conf_l.append(conf.item())
                pred_l.append(p.item())

            confidence_list.append(conf_l)
            prediction_list.append(pred_l)

            masked_ids = input_ids.clone()
            for k, _ in enumerate(masked_ids):
                try:
                    masked_ids[k][indices[k][m_idx+1]] = self.mask_idx
                except:
                    print("Skipped")
                    continue

        output = {'prediction_list': prediction_list, 'indices': indices, 'confidence': np.array(confidence_list)}

        return output

    def forward(self, input_ids, attention_mask, labels=None):
        """
        - x: input
        """
        emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)
        logits = self.cls(emb_x) # output from a FC layer
        

        if labels is not None:

            loss_fn = nn.CrossEntropyLoss()
            loss_ce = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

            output = {'logits': logits, 'loss': loss_ce, 'emb': emb_x}

            return output

        output = {'logits': logits, 'emb': emb_x}

        return output

