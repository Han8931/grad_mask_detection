import torch
import torch.nn.functional as F
from transformers import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers import get_linear_schedule_with_warmup

from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset

import pdb
import numpy as np
import random
import pandas as pd
import os, sys

from model.train import LinearScheduler, batch_len
from model.model_adv_multi import *

from utils.utils import boolean_string, print_args, save_checkpoint, model_eval, load_checkpoint, model_evaluation
from utils.dataloader import trans_dataloader
import utils.logger as logger
import datetime
import argparse


def get_parser(model_type: str):
    parser = argparse.ArgumentParser(description='RL based Adv Attack ')
    parser.add_argument('--exp_dir', type=str, default="./experiments/cls/", help='Experiment directory.')
    parser.add_argument('--exp_msg', type=str, default="CLS Transformer", help='Simple log for experiment')
    parser.add_argument('--gpu_idx', type=int, default=10, help='GPU Index')

    parser.add_argument('--eval', type=boolean_string, default=False, help='Evaluation')
    parser.add_argument('--model_dir_path', default='./', type=str, help='Save Model dir')
    parser.add_argument('--save_model', default='cls_trans', type=str, help='Save Model name')
    parser.add_argument('--load_model', default='cls_trans', type=str, help='Model name')
    parser.add_argument('--save', type=boolean_string, default=True, help='Evaluation')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    # Training
    parser.add_argument('--epochs', type=int, default=10, help='Training Epochs')
    parser.add_argument('--num_mask', type=int, default=1, help='Number of masking')

    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--model', default='distil', type=str, help='model',
                        choices=['albert', 'bert', 'rnn', 'distil', 'roberta', 'roberta-large'])
    parser.add_argument('--optim', default='adamw', type=str, help='optimizer',
                        choices=['noam', 'adagrad', 'adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', default='linear', type=str, help='optimizer',
                        choices=['linear'])
    parser.add_argument('--lr', default=0.0001, type=float, help='Agent learning rate')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--clip', default=1.0, type=float, help='Clip grad')
    parser.add_argument('--top_k', type=int, default=3, help='Random sampling from a distribution')
    parser.add_argument('--p_vocab', default=0.1, type=float, help='Percentage of vocab')

    parser.add_argument('--embed_dim', type=int, default=768, help='LSTM Input_Dim')

    # Dataset
    parser.add_argument('--dataset', default='mr', type=str, help='Dataset', 
            choices=['ag', 'mr', 'imdb', 'mnli', 'sst', 'clinc', 'clinc_ood', 'clinc_oop', 'clinc_small', 'yelp', 'wnli'])
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--pad_idx', type=int, default=0, help='Padding idx')
    parser.add_argument('--mask_idx', type=int, default=103, help='[MASK] idx')

    args = parser.parse_known_args()[0]

    return args

args = get_parser("DistilBERT") 
print_args(args)

print("Setup Logger...")
now = datetime.datetime.now()
args.exp_dir = args.exp_dir + f"{now.year}_{now.month}_{now.day}/"
print(f"Experiment Dir: {args.exp_dir} || {args.save_model}-------------------")

log_path = logger.log_path(now, args.exp_dir, args.save_model)
exp_log = logger.setup_logger('perf_info', log_path)

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


print(f"Load Tokenizer...")
if args.model == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
elif args.model == 'roberta':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
elif args.model == 'roberta-large':
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
elif args.model == 'distil':
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
args.pad_idx = tokenizer.pad_token_id
args.mask_idx = tokenizer.mask_token_id

print(f"Tokenizer: {args.model} || PAD: {args.pad_idx} || MASK: {args.mask_idx}") 
#stop_ids = stopwords_id(tokenizer)
#stop_ids = torch.Tensor(stop_ids).to(args.device)


train_dataloader, test_dataloader = trans_dataloader(args.dataset, tokenizer, args)

if args.dataset == 'ag':
    print(f"Load AGNews Dataset...")
    args.num_classes = 4

elif args.dataset == 'mr':
    print(f"Load Movie Review Dataset...")
    args.num_classes = 2

elif args.dataset == 'sst':
    print(f"Load SST-2 Dataset...")
    args.num_classes = 2

elif args.dataset == 'imdb':
    print(f"Load IMDB Dataset...")
    args.num_classes = 2

elif args.dataset == 'mnli':
    print(f"Load MNLI Dataset...")
    args.num_classes = 3

elif args.dataset == 'yelp':
    print(f"Load Yelp Dataset...")
    args.num_classes = 2

elif args.dataset == 'wnli':
    print(f"Load WNLI Dataset...")
    args.num_classes = 2

else:
    print("Classification task must be either ag or mr")

train_niter = len(train_dataloader)
total_iter = len(train_dataloader) * args.epochs

# Create Model 
print(f"Load Model...")

if args.model == 'bert':
    encoder = BertModel.from_pretrained('bert-base-uncased')
    Cls = SeqClsModel(input_size=768, output_size=args.num_classes, dropout=args.dropout)

elif args.model == 'roberta':
    encoder = RobertaModel.from_pretrained('roberta-base')
    encoder.config.num_labels = args.num_classes
    print(f"Encoder: {encoder.config.num_labels}") 
    Cls = RobertaClassificationHead(encoder.config)

elif args.model == 'roberta-large':
    encoder = RobertaModel.from_pretrained('roberta-large')
    encoder.config.num_labels = args.num_classes
    print(f"Encoder: {encoder.config.num_labels}") 
    Cls = RobertaClassificationHead(encoder.config)

elif args.model == 'distil':
    encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
    Cls = SeqClsModel(input_size=768, output_size=args.num_classes, dropout=args.dropout)

Enc = Encoder(encoder)
model = ClsText(Enc, Cls, args=args)
model.to(args.device)
model.train()

optimizer = AdamW(model.parameters(), lr=args.lr)

print("Start Training...")
best_acc = 0
best_epoch = 0

logger.args_logger(args, args.exp_dir)

for epoch in range(args.epochs):
    model.train()
    loss_epoch = []
    loss_ood_epoch = []

    for batch_idx, batch in enumerate(train_dataloader):

        optimizer.zero_grad()           

        input_ids = batch['input_ids'].to(args.device)
        attention_mask = batch['attention_mask'].to(args.device)
        labels = batch['labels'].to(args.device) 
        b_length = batch_len(input_ids, args.pad_idx)

        output = model(input_ids, attention_mask, labels)

        loss = output['loss']
        loss.backward()    
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()   
        loss_epoch.append(loss.item())

        if batch_idx % 100 == 0:
            log = f"Epoch: {epoch} || Iter: {batch_idx} || Loss: {np.mean(loss_epoch[-100:]):.3f}"
            print(log, flush=True)
            exp_log.info(log)

        curr = epoch * train_niter + batch_idx
        LinearScheduler(optimizer, total_iter, curr, args.lr)

    log = f"\nEpoch: {epoch} || Loss: {np.mean(loss_epoch):.3f}"
    print(log, flush=True)
    exp_log.info(log)

    if args.dataset=='mnli':
        acc_id = model_evaluation(model, test_id_dataloader, args)
        acc_ood = model_evaluation(model, test_ood_dataloader, args)

        if best_acc<acc_id:
            best_acc=acc_id
            best_epoch=epoch
        log = f"Epoch: {epoch} || TestID Acc: {acc_id:.4f} || TestOOD Acc: {acc_ood:.4f} || BestIDAcc: {best_acc:.4f} || BestEpoch: {best_epoch}"
        print(log, flush=True)
        exp_log.info(log)

    else:
        acc = model_evaluation(model, test_dataloader, args)
    
        if best_acc<acc:
            best_acc=acc
            best_epoch=epoch
            if args.save:
                save_checkpoint(args.save_model, model, epoch, ckpt_dir=args.model_dir_path)

        log = f"Epoch: {epoch} || Test Acc: {acc:.4f} || BestAcc: {best_acc:.4f} || BestEpoch: {best_epoch}"
        print(log, flush=True)
        exp_log.info(log)

print("End Training...")


