import torch
import time, datetime
import os, sys
import matplotlib.pyplot as plt

from collections import Counter
from model.train import batch_len

from sklearn.metrics import f1_score as f1_score_sk
import sklearn.metrics as metrics

import numpy as np
import pickle, json
import pdb

def max_min_scaler(pred):
    min_pred = pred.min()
    max_pred = pred.max()
    pred_norm = (pred-min_pred)/(max_pred-min_pred)
    return pred_norm

def search_threshold(stat, end, step, up=True, scale=True):
    pred = np.array(stat.pred)
    y_true = pred[:,1].astype(int)

    feature = max_min_scaler(pred[:,0])

    best_f1 = 0
    best_t = 0

    for threshold in np.arange(0, end, step):
        if up:
            y_pred = feature>threshold
        else:
            y_pred = feature<threshold
        f1_macro = f1_score_sk(y_true, y_pred, average='macro')
        if f1_macro>best_f1:
            best_f1 = f1_macro
            best_t = threshold
    return best_f1, best_t

def match_string(string1, string2):
    tokens1 = string1.split()
    tokens2 = string2.split()


    new_string = ''

    for i, t2 in enumerate(tokens2):

        try:
            #print(t2, tokens1[i])
            if tokens1[i]==t2:
                new_string += (t2 + ' ')
            elif t2 in count1.keys() and count1[t2]==count2[t2]:
                new_string += (t2 + ' ')
            else:
                new_string += red(t2)
                #new_string += ('*' + t2 + '* ')
        except:
            new_string += red(t2)
#            new_string += ('*' + t2 + '* ')

    return new_string



def count_diff(txt1, txt2):
    """txt1: orig, txt2: pert"""

    if type(txt1)==str:
        txt1 = txt1.split(" ")
        txt2 = txt2.split(" ")

    count1 = Counter(txt1)
    count2 = Counter(txt2)

    diff = count2-count1
    diff_vals = list(diff.values())
    diff_sum = sum(diff_vals)

    txt1_len = len(txt1)

    return diff_sum, txt1_len

def text_comparison(orig, pert):
    count = []
    length = []
    for org, prt in zip(orig, pert):
        count_, len_ = count_diff(org, prt)
        count.append(count_)
        length.append(len_) 
    return np.array(count), np.array(length)

def model_evaluation(model, test_dataloader, args):
    model.eval()
    TP = 0
    n_samples = len(test_dataloader.dataset)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):

            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device) 

            output = model(input_ids, attention_mask, labels)
            #output = model.grad_mask(input_ids, attention_mask, labels)
            preds = output['logits']
            correct = preds.argmax(dim=-1).eq(labels)
            TP += correct.sum().item()

        acc = 100*(TP/n_samples)

    return acc

def model_eval(model, test_iterator, device):
    model.eval()
    n_correct = 0.
    n_samples = len(test_iterator.dataset)
    confidence = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iterator):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)

            p_out = outputs['logits'].softmax(dim=-1)
            conf_ = p_out.gather(1, labels.unsqueeze(1))
            confidence.extend(conf_.squeeze(1).tolist())

            preds = outputs['logits'].argmax(dim=-1)
            correct = preds.eq(labels)
            n_correct += correct.sum().item()

    acc = 100*(n_correct/n_samples)
    print(f"Test Acc: {acc:.4f} || Confidence: {np.mean(confidence)*100:.2f}")


def save_dataset(dataset, directory, dataset_name):
    # Pickle format version 4.0
    path = os.path.join(directory, dataset_name+'.pickle')
    if not os.path.isdir(directory):
        os.makedirs(directory)

    with open(path, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dataset(dataset, directory, dataset_name):
    # Pickle format version 4.0
    path = os.path.join(directory, dataset_name)
    with open(path, 'rb') as handle:
        dataset = pickle.load(handle)
    return dataset

def save_checkpoint(save_model, model, epoch, ckpt_dir):
    ckpt_name = save_model + f"_{epoch}"
    print(f"Save: {ckpt_name}")
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
    }
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save(state, os.path.join(ckpt_dir, ckpt_name))
    #mon.dump('checkpoint.pt', state, method='torch', keep=5)  # keep only 5 recent checkpoints


def load_checkpoint(model, model_name, ckpt_dir):
    print(f"Load: {model_name}") 
    load_path = os.path.join(ckpt_dir, model_name)
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    return model

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def print_args(args):
    """
    Print all arguments in argparser
    """

    if args.device == "cuda":
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)

    print("---------------------------------------------------------")
    print("---------------------------------------------------------")
    now = datetime.datetime.now()
    print(f"||Experiment Date:{now.year}-{now.month}-{now.day}||")

    print("Arguments List: \n")
    if args.device == "cuda":
        print(f"- Running on GPU: {gpu_name} || GPU Idx: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print(f"- Running on CPU")
    for arg in vars(args):
        print(f"- {arg}: {getattr(args, arg)}")

    print("---------------------------------------------------------")
    print("---------------------------------------------------------\n")


