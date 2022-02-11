"""
Adversarial example generation scripts with TextAttack_v2
"""
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import transformers
from transformers import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertForSequenceClassification, BertModel

import pdb
import numpy as np
import random
import os, sys
import argparse
import pandas as pd

from textattack.datasets import HuggingFaceDataset
from textattack.transformations import WordSwap
from textattack.search_methods import GreedySearch
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.goal_functions import UntargetedClassification
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult, SkippedAttackResult

from textattack.loggers import CSVLogger
from model.textattack_model import CustomWrapper, print_function

from model.model_adv_multi import *
import types

from utils.utils import boolean_string, print_args, save_checkpoint, load_checkpoint, model_eval
from utils.dataloader import text_dataloader

from datetime import timedelta
import time, datetime

def get_parser(model_type: str):
    parser = argparse.ArgumentParser(description='RL based Adv Attack ')

    parser.add_argument('--exp_dir', type=str, default="./experiments/cls/", help='Experiment directory.')
    parser.add_argument('--exp_msg', type=str, default="CLS Transformer", help='Simple log for experiment')

    # parser.add_argument('--deterministic', type=boolean_string, default=True, help='Deterministic')
    parser.add_argument('--eval', type=boolean_string, default=False, help='Evaluation')
    parser.add_argument('--model_dir_path', default='./', type=str, help='Save Model dir')
    parser.add_argument('--load_model', default='cls_attn_3', type=str, help='Model name')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    # Training
    parser.add_argument('--epochs', type=int, default=5, help='Training Epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--model', default='distil', type=str, help='model', 
            choices=['bert', 'distil', 'roberta', 'roberta-large'])
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--top_k', type=int, default=3, help='Random sampling from a distribution')
    parser.add_argument('--embed_dim', type=int, default=768, help='Attn Input_Dim')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Attn hidden_Dim')


    # Attack
    parser.add_argument('--attack_method', default='pwws', type=str, help='TextAttack Method',
                        choices=['pwws', 'textfooler', 'character', 'a2t', 'bae'])

    parser.add_argument('--multi_mask', type=int, default=1, help='Masking multiple token')

    # Dataset
    parser.add_argument('--save_data', type=boolean_string, default=True, help='Save data')
    parser.add_argument('--dataset', default='mr', type=str, help='Dataset', choices=['ag', 'mr', 'imdb', 'wnli', 'wgra', 'yelp', 'sst'])
    parser.add_argument('--dataset_type', default='test', type=str, help='Dataset', choices=['train', 'test'])
    parser.add_argument('--nth_data', type=int, default=0, help='Dataset idx')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--pad_idx', type=int, default=0, help='Padding idx')
    parser.add_argument('--mask_idx', type=int, default=103, help='[MASK] idx')

    parser.add_argument('--n_success', type=int, default=1000, help='Num Adv Success')
    parser.add_argument('--shuffle', type=boolean_string, default=True, help='Dataset Split')

    args = parser.parse_known_args()[0]

    return args

args = get_parser("Transformer") 

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

f_name = "_"+str(args.load_model)+"_"+args.dataset_type+".csv"
adv_path = os.path.join('./data/'+args.dataset+'_'+args.attack_method+f_name)

args.adv_path = adv_path

print("Load Dataset...") 
train, test = text_dataloader(args.dataset, args)
if args.dataset_type =='test':
    dataloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=args.shuffle, num_workers=0)
else:
    dataloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=args.shuffle, num_workers=0)

print_args(args)
print(f"Adv Path: {adv_path}") 
print(f"Save Data: {args.save_data}") 

print("----------------------------------")
print(f"Dataset Type: {args.dataset_type}") 
print("----------------------------------")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Model 
print(f"Load Model...")

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
args.cls_token = tokenizer.cls_token_id
args.sep_token = tokenizer.sep_token_id


if args.model == 'bert':
    print("Load BERT")
    encoder = BertModel.from_pretrained('bert-base-uncased')
    Cls = SeqClsModel2(input_size=768, output_size=args.num_classes, dropout=args.dropout)

elif args.model == 'roberta':
    print("Load Roberta")
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
    print("Load Distil")
    encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
    Cls = SeqClsModel(input_size=768, output_size=args.num_classes, dropout=args.dropout)

Enc = Encoder(encoder)
model = ClsText(Enc, Cls, args)

model = load_checkpoint(model, args.load_model, args.model_dir_path)
model.eval()

model_wrapper = CustomWrapper(model, tokenizer, args)
model_wrapper.model.to(device)

# Attack Recipe
if args.attack_method == 'pwws':
    from textattack.attack_recipes import PWWSRen2019
    attack = PWWSRen2019.build(model_wrapper)
elif args.attack_method == 'character':
    from textattack.attack_recipes.pruthi_2019 import Pruthi2019
    attack = Pruthi2019.build(model_wrapper)
elif args.attack_method == 'textfooler':
    from textattack.attack_recipes import TextFoolerJin2019
    attack = TextFoolerJin2019.build(model_wrapper)
elif args.attack_method == 'a2t':
    from textattack.attack_recipes.a2t_yoo_2021 import A2TYoo2021
    attack = A2TYoo2021.build(model_wrapper)
elif args.attack_method == 'bae':
    from textattack.attack_recipes.bae_garg_2019 import BAEGarg2019
    attack = BAEGarg2019.build(model_wrapper)

attack.goal_function.maximizable = False

print(attack)

logger = CSVLogger(color_method='html')

n_trial = 0
num_successes = 0
num_skipped = 0
num_failed = 0

total_queries = 0

df_adv = pd.DataFrame()

n_exception = 0
n_pred_error = 0

start_t_gen = time.perf_counter()



print("Start Attack...")
for batch_idx, batch in enumerate(dataloader):

    if num_successes>args.n_success-1:

        eval_t = time.perf_counter()-start_t_gen
        print_function(args, f_name, batch_idx, num_successes, num_failed, num_skipped)
        print(f"Total Elapsed Time: {timedelta(seconds=eval_t)}", flush=True) 

        if args.save_data == True:
            if not os.path.isdir('./data/'):
                os.makedirs('./data/')
            
            df_adv.to_csv(adv_path)
            print(f"Save data...: {adv_path}", flush=True) 

        exit(0)

    label = batch['label'].item() # Ground truth label
    orig = batch['text'][0] # Clen Text 

    result = attack.attack(orig, label)
    n_query = result.num_queries

    pert = result.perturbed_text()
    logger.log_attack_result(result)
    attack_result = logger.df.result_type.iloc[-1]


    if attack_result == 'Skipped':
        num_skipped+=1
        result_type = 'Skipped'
    elif attack_result == 'Successful':
        num_successes+=1
        result_type = 'Successful'
    elif attack_result == 'Failed':
        num_failed+=1
        result_type = 'Failed'
    else:
        continue

    adv_dict = {'pert': pert, 'orig': orig, 'ground_truth_output': label, 'result_type': result_type, 'n_query': n_query}
    df_adv = df_adv.append(adv_dict, ignore_index=True)

    if batch_idx%10==0:
        print(result.__str__(color_method='ansi'))
        print_function(args, f_name, batch_idx, num_successes, num_failed, num_skipped)

eval_t = time.perf_counter()-start_t_gen
print_function(args, f_name, batch_idx, num_successes, num_failed, num_skipped)
print(f"Total Elapsed Time: {timedelta(seconds=eval_t)}", flush=True) 

if args.save_data == True:
    if not os.path.isdir('./data/'):
        os.makedirs('./data/')
    
    df_adv.to_csv(adv_path)
    print(f"Save data...: {adv_path}", flush=True) 


