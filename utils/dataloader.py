import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.utils.data import TensorDataset
from datasets import load_dataset, Dataset, list_datasets

import numpy as np
import pandas as pd
import pdb
import string, operator
import re

from abc import *
import json

import random
import sys, os, pdb
import time, datetime
import argparse, copy

import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.corpus import wordnet
import nltk



class RawDataset(torch.utils.data.Dataset):
    def __init__(self, text_pair, labels):
        self.text_pair = text_pair
        self.labels = labels

    def __getitem__(self, idx):

        item = {"text": self.text_pair[idx]}
        #item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.text_pair)


def trans_detection_dataloader(df_adv, tokenizer, args):

    orig_label = df_adv['ground_truth_output']
    orig_label = list(orig_label.astype(int).values) # Orig text label

    adv_text = df_adv['pert']
    orig_text = df_adv['orig']

    text_orig = list(orig_text.values) # Orig text
    text_pert = list(adv_text.values) # Orig text

    text_pair = []
    for org, adv in zip(text_orig, text_pert):
        text_pair.append((org, adv))

    print(f"OrgExamples: {len(orig_text)}")
    print(f"AdvExamples: {len(adv_text)}")

    test_set = RawDataset(text_pair, orig_label)

    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    return test_dataloader

class ClsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings.input_ids)


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings.input_ids)


def trans_dataloader(dataset, tokenizer, args):

    if dataset == 'ag':
        """
        0: world
        1: sports
        2: buisiness
        3: Sci/Tech
        """
        dataset = load_dataset("ag_news")
        test, train = dataset['test'], dataset['train']

    elif dataset == 'wnli':
#        dataset = load_dataset("winograd_wsc", 'wsc273')
        train = pd.read_csv('./data/WNLI/train.tsv', delimiter = '\t')
        #dev = pd.read_csv('./data/WNLI/dev.tsv', delimiter = '\t')
        test = pd.read_csv('./data/WNLI/dev.tsv', delimiter = '\t')

        train_txt_1 = list(train['sentence1'])
        train_txt_2 = list(train['sentence2'])
        train_label = list(train['label'])

        test_txt_1 = list(test['sentence1'])
        test_txt_2 = list(test['sentence2'])
        test_label = list(test['label'])

        train_data = tokenizer(train_txt_1, train_txt_2, padding=True, truncation=True, max_length=256)
        test_data = tokenizer(test_txt_1, test_txt_2, padding=True, truncation=True, max_length=256)

        #train_label = [int(i) for i in train_label]

        train_set = ClsDataset(train_data, train_label)
        test_set = ClsDataset(test_data, test_label)

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    elif dataset == 'mr':
        """
        Sentiment polarity datasets: binary 0/1
        """
        dataset = load_dataset("rotten_tomatoes")
        test, train = dataset['test'], dataset['train']

        train_data = tokenizer(train['text'], padding=True, truncation=True)
        test_data = tokenizer(test['text'], padding=True, truncation=True)
        # dev_data = tokenizer(dev['text'], padding=True, truncation=True)

        train_label = train['label']
        test_label = test['label']
        # dev_label = dev['label']

        train_set = ClsDataset(train_data, train_label)
        test_set = ClsDataset(test_data, test_label)

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    elif dataset == 'mnli':
        dataset = load_dataset("multi_nli")
        train, test_id, test_ood = dataset['train'], dataset['validation_matched'], dataset['validation_mismatched']

        train_data = tokenizer(train['premise'], train['hypothesis'], padding=True, truncation=True)
        test_id_data = tokenizer(test_id['premise'], test_id['hypothesis'], padding=True, truncation=True)
        test_ood_data = tokenizer(test_ood['premise'], test_ood['hypothesis'], padding=True, truncation=True)

        train_label = train['label']
        test_id_label = test_id['label']
        test_ood_label = test_ood['label']

        train_set = NLIDataset(train_data, train_label)
        test_id_set = NLIDataset(test_id_data, test_id_label)
        test_ood_set = NLIDataset(test_ood_data, test_ood_label)

        train_dataloader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True)
        test_id_dataloader = DataLoader(test_id_set, batch_size = args.batch_size, shuffle=True)
        test_ood_dataloader = DataLoader(test_ood_set, batch_size = args.batch_size, shuffle=True)

        return train_dataloader, test_id_dataloader, test_ood_dataloader

    elif dataset == 'sst':
        """
        Stanford Sentiment Treebank polarity datasets: binary 0:neg/1:pos
        """
        dataset = load_dataset("sst")
        test, train = dataset['test'], dataset['train']

        train = pd.read_csv('./data/train.tsv', delimiter = '\t')
        train_txt = list(train['sentence'])
        train_label = list(train['label'])
        #train_label = [int(i) for i in train_label]

        test_label = test['label']
        test_label = np.array(test_label)
        pos_idx = test_label>0.6
        neg_idx = test_label<0.4

        pos_label = np.ceil(test_label[pos_idx]).astype(int)
        neg_label = np.floor(test_label[neg_idx]).astype(int)

        test_txt = np.array(test['sentence'])
        pos_txt = test_txt[pos_idx]
        neg_txt = test_txt[neg_idx]

        # Test
        test_txt = list(pos_txt)+list(neg_txt)
        test_label = list(pos_label)+list(neg_label)
        test_label = [int(i) for i in test_label]

        train_data = tokenizer(train_txt, padding=True, truncation=True)
        test_data = tokenizer(test_txt, padding=True, truncation=True)

        train_set = ClsDataset(train_data, train_label)
        test_set = ClsDataset(test_data, test_label)

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    elif dataset == 'imdb':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        dataset = load_dataset("imdb")
        test, train = dataset['test'], dataset['train']

    elif dataset == 'yelp':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        dataset = load_dataset("yelp_polarity")
        test, train = dataset['test'], dataset['train']

    else:
        print("Specift dataset correctly")
        exit(0)

    print(f"Trainset Size: {len(train)}")
    print(f"Testset Size: {len(test)}")

    train_data = tokenizer(train['text'], padding=True, truncation=True)
    test_data = tokenizer(test['text'], padding=True, truncation=True)
    # dev_data = tokenizer(dev['text'], padding=True, truncation=True)

    train_label = train['label']
    test_label = test['label']
    # dev_label = dev['label']

    train_set = ClsDataset(train_data, train_label)
    test_set = ClsDataset(test_data, test_label)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def text_dataloader(dataset, args):

    if dataset == 'ag':
        """
        0: world
        1: sports
        2: buisiness
        3: Sci/Tech
        """
        dataset = load_dataset("ag_news")
        test, train = dataset['test'], dataset['train']
        args.num_classes = 4

    elif dataset == 'mr':
        """
        Sentiment polarity datasets: binary 0/1
        """
        dataset = load_dataset("rotten_tomatoes")
        test, train = dataset['test'], dataset['train']
        args.num_classes = 2

    elif dataset == 'sst':
        """
        Stanford Sentiment Treebank polarity datasets: binary 0:neg/1:pos
        """
        args.num_classes = 2
        dataset = load_dataset("sst")
        test, train = dataset['test'], dataset['train']

        train = pd.read_csv('./data/train.tsv', delimiter = '\t')
        train_txt = list(train['sentence'])
        train_label = list(train['label'])
        #train_label = [int(i) for i in train_label]

        test_label = test['label']
        test_label = np.array(test_label)
        pos_idx = test_label>0.6
        neg_idx = test_label<0.4

        pos_label = np.ceil(test_label[pos_idx]).astype(int)
        neg_label = np.floor(test_label[neg_idx]).astype(int)

        test_txt = np.array(test['sentence'])
        pos_txt = test_txt[pos_idx]
        neg_txt = test_txt[neg_idx]

        # Test
        test_txt = list(pos_txt)+list(neg_txt)
        test_label = list(pos_label)+list(neg_label)
        test_label = [int(i) for i in test_label]

        return train_txt, test_txt, train_label, test_label

    elif dataset == 'imdb':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        args.num_classes = 2
        dataset = load_dataset("imdb")
        test, train = dataset['test'], dataset['train']


    else:
        print("Specift dataset correctly")
        exit(0)

    print(f"Trainset Size: {len(train)}")
    print(f"Testset Size: {len(test)}")


    return train, test

