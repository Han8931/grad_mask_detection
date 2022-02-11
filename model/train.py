import torch
from transformers import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import string

import nltk

import pdb

def LinearScheduler(optimizer, total_iter, curr, lr_init):
    lr = -(lr_init / total_iter) * curr + lr_init
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def batch_len(input_ids, pad_idx=0):
    b_length = (input_ids != pad_idx).data.sum(dim=-1)
    return b_length


