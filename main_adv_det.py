import torch
import torch.nn.functional as F

from transformers import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertForSequenceClassification, BertModel

from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

import pdb
import numpy as np
import random
import pandas as pd
import os, sys
from collections import Counter

from model.train import LinearScheduler, batch_len
from model.model_adv_multi import *
from model.robust_train import DetectionStat

from utils.utils import boolean_string, print_args, load_checkpoint
from utils.dataloader import trans_detection_dataloader
import utils.logger as logger
import datetime
# from utils.task_eval import 
import argparse

from collections import Counter

from datetime import timedelta
import time, datetime

from tqdm import tqdm

import re

def get_parser(model_type: str):
    parser = argparse.ArgumentParser(description='Detection')
    parser.add_argument('--exp_dir', type=str, default="./experiments/cls/", help='Experiment directory.')
    parser.add_argument('--exp_msg', type=str, default="CLS Transformer", help='Simple log for experiment')
    parser.add_argument('--gpu_idx', type=int, default=10, help='GPU Index')

    # parser.add_argument('--deterministic', type=boolean_string, default=True, help='Deterministic')
    parser.add_argument('--eval', type=boolean_string, default=False, help='Evaluation')
    parser.add_argument('--model_dir_path', default='./', type=str, help='Save Model dir')
    parser.add_argument('--save_model', default='cls_trans', type=str, help='Save Model name')
    parser.add_argument('--load_model', default='cls_trans', type=str, help='Model name')
    parser.add_argument('--adv_path', default=0.1, type=float, help='adv_path')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    # Training
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--model', default='distil', type=str, help='model', choices=['bert', 'distil', 'roberta'])
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--clip', default=1.0, type=float, help='Clip grad')
    parser.add_argument('--top_p', type=float, default=0.05, help='Random sampling from a distribution')

    parser.add_argument('--embed_dim', type=int, default=768, help='LSTM Input_Dim')

    # Inference
    parser.add_argument('--multi_mask', type=int, default=1, help='Masking multiple token')
    parser.add_argument('--conf_feature', type=str, default='conf_sub_square', help='Confidence feature type')
    parser.add_argument('--dataset_type', default='test', type=str, help='Dataset', choices=['train', 'test'])

    # Dataset
    parser.add_argument('--dataset', default='mr', type=str, help='Dataset', 
            choices=['ag', 'mr', 'imdb', 'sst', 'wgra', 'yelp', 'mnli_id', 'mnli_ood'])
    parser.add_argument('--nth_data', type=int, default=0, help='Dataset idx')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--pad_idx', type=int, default=0, help='Padding idx')
    parser.add_argument('--mask_idx', type=int, default=103, help='[MASK] idx')
    parser.add_argument('--mask_p', type=float, default=0.3, help='[MASK] idx')

    parser.add_argument('--attack_method', default='pwws', type=str, help='TextAttack Method',
                        choices=['pwws', 'textfooler', 'character', 'a2t', 'bae'])


    args = parser.parse_known_args()[0]

    return args

args = get_parser("DistilBERT") 
print_args(args)

print("Setup Logger...")
now = datetime.datetime.now()
args.exp_dir = args.exp_dir + f"{now.year}_{now.month}_{now.day}/"
print(f"Experiment Dir: {args.exp_dir} || {args.load_model}-------------------")

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


print(f"Load Tokenizer...")
if args.model == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    args.pad_idx = tokenizer.pad_token_id
    args.mask_idx = tokenizer.mask_token_id
elif args.model == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    args.pad_idx = tokenizer.pad_token_id
    args.mask_idx = tokenizer.mask_token_id
elif args.model == 'distil':
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    args.pad_idx = tokenizer.pad_token_id
    args.mask_idx = tokenizer.mask_token_id
else:
    print("Specify model correct...")

if args.dataset=='mnli_id':
    f_name = f"mnli_{args.attack_method}_{args.load_model}_{args.dataset_type}_id.csv"
elif args.dataset=='mnli_ood':
    f_name = f"mnli_{args.attack_method}_{args.load_model}_{args.dataset_type}_ood.csv"
else:
    f_name = f"{args.dataset}_{args.attack_method}_{args.load_model}_{args.dataset_type}.csv"

adv_path = os.path.join('./data/'+f_name)

print(f"AdvPath: {adv_path}", flush=True) 

df = pd.read_csv(adv_path)
df_ = df[df['result_type']!='Skipped'].reset_index()
df_ = df_[df_['result_type']!='Failed'].reset_index()

n_skip = (df['result_type']=='Skipped').sum()
n_fail = (df['result_type']=='Failed').sum()
print(f"# Samples: {df.shape[0]} || # Skip: {n_skip} || # Fail: {n_fail}", flush=True) 

test_dataloader = trans_detection_dataloader(df_, tokenizer, args)

# Load Dataset
if args.dataset == 'ag':
    print(f"Load AGNews Dataset...")
    args.num_classes = 4

elif args.dataset == 'imdb':
    print(f"Load IMDB Dataset...")
    args.num_classes = 2

elif args.dataset == 'mr':
    print(f"Load MR Dataset...")
    args.num_classes = 2

elif args.dataset == 'wgra':
    print(f"Load WinoGrande Dataset...")
    args.num_classes = 2

elif args.dataset == 'sst':
    print(f"Load SST-2 Dataset...")
    args.num_classes = 2

elif args.dataset == 'yelp':
    print(f"Load Yelp Dataset...")
    args.num_classes = 2

elif args.dataset == 'mnli_id' or args.dataset == 'mnli_ood' :
    print(f"Load MNLI Dataset...")
    args.num_classes = 3

else:
    print("Speficy Dataset Correctly...")

# Create Model 
print(f"Load Model...")

if args.model == 'roberta':
    encoder = RobertaModel.from_pretrained('roberta-base')
    encoder.config.num_labels = args.num_classes
    print(f"Encoder: {encoder.config.num_labels}") 
    Cls = RobertaClassificationHead(encoder.config)

if args.model == 'bert':
    encoder = BertModel.from_pretrained('bert-base-uncased')
    Cls = SeqClsModel(input_size=768, output_size=args.num_classes, dropout=args.dropout)

elif args.model == 'roberta-large':
    encoder = RobertaModel.from_pretrained('roberta-large')
    encoder.config.num_labels = args.num_classes
    print(f"Encoder: {encoder.config.num_labels}") 
    Cls = RobertaClassificationHead(encoder.config)

elif args.model == 'distil':
    encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
    Cls = SeqClsModel(input_size=768, output_size=args.num_classes, dropout=args.dropout)
else:
    print("Specify model correctly...") 

Enc = Encoder(encoder)
model = ClsTextDet(Enc, Cls, args=args)
model = load_checkpoint(model, args.load_model, args.model_dir_path)
model.to(args.device)
model.eval()

print("--------------------------------------------------------------------------------------------------------------------------")
print("--------------------------------------------------------------------------------------------------------------------------")
print(f"Start Evaluation...", flush=True) 
print(f"Model: {args.load_model} || Dataset: {adv_path} || Top-P: {args.top_p}", flush=True) 
print(f"ConfFeature: {args.conf_feature} || Multi-Mask: {args.multi_mask}", flush=True)
print("--------------------------------------------------------------------------------------------------------------------------")
print("--------------------------------------------------------------------------------------------------------------------------")


txt_length_list = []
stat = DetectionStat()
start_t = time.perf_counter()
adv_token_result = []

def tokenize_function(text, tokenizer):
    encoded = tokenizer(text, add_special_tokens=True, padding="max_length", 
            max_length=512, truncation=True, return_tensors='pt')
    return encoded

def detection_function(model, input_ids, attention_mask, conf_org, pred, args):
    if args.conf_feature != "org":
        indices, _ = model.grad_detection_batch2(input_ids, attention_mask, topk=5, pred=pred) 
        output = model.iterative_grad_mask_detection_batch(input_ids, attention_mask, pred_org=pred, 
                topk=args.top_p, indices=indices, multi_mask=args.multi_mask)
        confidence = output['confidence'] 

    val = (conf_org-confidence.min(axis=0))**2  

    return val

N_fail = 0
N_skip = 0

for batch_idx, batch in enumerate(test_dataloader):

    text = batch['text']
    labels = batch['labels'].to(args.device)

    c_txt = text[0]
    a_txt = text[1]

    orig_enc = tokenize_function(c_txt, tokenizer)
    pert_enc = tokenize_function(a_txt, tokenizer)

    input_ids = orig_enc['input_ids'].to(args.device)
    attention_mask = orig_enc['attention_mask'].to(args.device)

    input_ids_adv = pert_enc['input_ids'].to(args.device)
    attention_mask_adv = pert_enc['attention_mask'].to(args.device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        preds = output['logits']
        pred_orig = preds.argmax(dim=-1)
        conf_org = preds.softmax(dim=-1).max(dim=1)[0] 
        conf_org = conf_org.detach().cpu().numpy()

    with torch.no_grad():
        output = model(input_ids_adv, attention_mask_adv)
        preds_adv = output['logits']
        pred_adv = preds_adv.argmax(dim=-1)
        conf_org_adv = preds_adv.softmax(dim=-1).max(dim=1)[0] 
        conf_org_adv = conf_org_adv.detach().cpu().numpy()

    pred_consistency = pred_orig.eq(labels)
    pred_consistency.sum()

    N_skip+=(labels.shape[0]-pred_consistency.sum()).item()
    
    attack_result = pred_orig.eq(pred_adv)
    N_fail+=attack_result.sum().item()

    val_c = detection_function(model, input_ids, attention_mask, conf_org, pred_orig, args)
    val_a = detection_function(model, input_ids_adv, attention_mask_adv, conf_org_adv, pred_adv, args)

    c_labels = torch.zeros(val_c.shape[0]).long()
    a_labels = torch.ones(val_c.shape[0]).long()

    stat.update2(val_c, c_labels)
    stat.update2(val_a, a_labels)

eval_t = time.perf_counter()-start_t
print(f"Elapsed Time ID: {timedelta(seconds=eval_t)}", flush=True) 

log = f"#_Adv: {df_.shape[0]} #_Skip: {N_skip} || # Fail: {N_fail}"
print(log, flush=True)

stat.print_perf(args)

print(" ")
print("="*40)
print(" ")

print("End Evaluation...", flush=True)


