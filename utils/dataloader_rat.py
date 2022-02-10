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

def clean_str(string, tokenizer=None):
    """
    Parts adapted from https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/mydatasets.py
    """
    assert isinstance(string, str)
    string = string.replace("<br />", "")
    string = re.sub(r"[^a-zA-Z0-9.]+", " ", string)

    return (
        string.strip().lower().split()
        if tokenizer is None
        else [t.text.lower() for t in tokenizer(string.strip())]
    )

def word_counter_fn(dataset_text):
    """
    Count whole words in the path
    """
    word_count = {}
    for sent in dataset_text:
        sent_tok = simple_tokenizer(sent)
        for word in sent_tok:
            if word not in word_count:
                word_count[word]=1
            else:
                word_count[word]+=1

    return word_count

def simple_tokenizer(sent):
    """
    This is a simple tokinizer
    Tokenize based on "space"

    Remove <s>, </s>
    """
    #tokenized = sent.strip().split(" ")
    tokenized = word_tokenize(sent)
    return tokenized

def sent2idx_mask(train_vocab, sent, tokenizer):
    #sent = sent.split(" ")
    sent = word_tokenize(sent)
    for i, word in enumerate(sent):
        if word not in train_vocab:
            sent[i] = "[MASK]"

    sent_mask = " ".join(sent)
    return sent_mask

def adv_dataloader(df_adv, tokenizer, args):
    text = list(df_adv['pert'].values)
    label = df_adv['ground_truth_output'].values

    out = tokenizer(text)
    text_ = tokenizer.batch_decode(out['input_ids'], skip_special_tokens=True)
    data = tokenizer(text_, padding=True, truncation=True)

    dataset = AdvDataset(data, label)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    return dataloader


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


class ClsDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, ood_label):
        self.encodings = encodings
        self.labels = labels
        self.ood_label = ood_label

    def __getitem__(self, idx):

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['idx'] = idx
        item['ood_label'] = torch.tensor(self.ood_label[idx])

        return item


    def __len__(self):
        return len(self.encodings.input_ids)



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

class ClsDataset2(torch.utils.data.Dataset):
    def __init__(self, encodings, encodings2, labels):
        self.encodings = encodings
        self.encodings2 = encodings2
        self.labels = labels

    def __getitem__(self, idx):
        
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['input_ids2'] = torch.tensor(self.encodings2['input_ids'][idx])
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

def preprocess_html(example):
    """
    Filter short sentences
    """
    #input_size/2
    txt_ = example['text']
    #p = re.compile(r'<.*?>')
    example['text'] = re.sub(r"<br /><br />", " ", txt_)

    return example

def preprocess2(example):
    stop_words = stopwords.words('english')

    text = "Nick likes to play football, however he is not too fond of tennis."

    text_tokens = word_tokenize(text)
    punctuation = string.punctuation

    tokens_without_sw = []
    tokens_sw = []
    for word in text_tokens:
        if word in stop_words or word in punctuation:
            # tokens_sw.append("[MASK]")
            tokens_sw.append(word)
        else:
            tokens_without_sw.append(word)

    # tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    # tokens_sw = [word for word in text_tokens if word in stopwords.words()]
    print(tokens_sw)
    print(tokens_without_sw)

    txt_ = example['text']
    txt_ = txt_.translate(str.maketrans('', '', string.punctuation))
    txt_ = re.sub(r"  ", " ", txt_)
    txt_ = txt_.strip() + "."
    example['text'] = txt_
    return example

def unk_tokenizer_setup(dataset_text, p_vocab):
    word_count = word_counter_fn(dataset_text)
    vocab = build_vocab(word_count, ratio=p_vocab)
    #word2idx, idx2word = word_dictionary(vocab)

    return vocab
    #return vocab, word2idx, idx2word


def dataset_mask(dataset_text, train_vocab, tokenizer):
    for i, sent in enumerate(dataset_text):
        sent_mask = sent2idx_mask(train_vocab, sent, tokenizer)
        dataset_text[i] = sent_mask

    return dataset_text

def build_dataset_vocab(dataset, tokenizer, args):
    if dataset == 'ag':
        """
        0: world
        1: sports
        2: buisiness
        3: Sci/Tech
        """
        dataset = load_dataset("ag_news")
        train = dataset['train']

        if args.split == True:
            split = train.train_test_split(test_size=0.3)
            train = split['test']  # We want to 30% of the original trainset, so take 'test'

        print(f"Trainset Size: {len(train)}")

        if args.prep:
            train = train.map(preprocess3)

        v_builder = Vocab_builder(train['text'], tokenizer)
        p_vocab, _ = v_builder.build_vocab(args.p_vocab)

        return p_vocab

    elif dataset == 'mr':
        """
        Sentiment polarity datasets: binary 0/1
        """

        dataset = load_dataset("rotten_tomatoes")
        train = dataset['train']

        if args.min_length_filter == True:
            sample_mask_train = sample_mask_fn(train)

        print(f"Trainset Size: {len(train)}")

        if args.prep:
            train = train.map(preprocess)


        v_builder = Vocab_builder(train['text'], tokenizer)
        p_vocab, _ = v_builder.build_vocab(args.p_vocab)
        return p_vocab

    elif dataset == 'imdb':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        dataset = load_dataset("imdb")
        train = dataset['train']

        if args.split == True:
            split_train = train.train_test_split(test_size=0.4)
            train = split_train['test']  # We want to 30% of the original trainset, so take 'test'

        if args.min_length_filter == True:
            sample_mask_train = sample_mask_fn(train)

        print(f"Trainset Size: {len(train)}")

        if args.prep:
            train = train.map(preprocess_html)

        v_builder = Vocab_builder(train['text'], tokenizer)
        p_vocab, _ = v_builder.build_vocab(args.p_vocab)
        return p_vocab

    else:
        print("Set dataset corrrectly")

def adv_data_processor2(df_adv, tokenizer, args, train=True):
    text_adv = list(df_adv['pert'].values)
    label_adv = df_adv['ground_truth_output'].values
    label_adv = list(label_adv.astype(int))

    out = tokenizer(text_adv)
    text_adv = tokenizer.batch_decode(out['input_ids'], skip_special_tokens=True)
    #data_adv = tokenizer(text_adv, padding=True, truncation=True)

    #return text_adv, label_adv
    return (text_adv_orig, label_adv), (text_adv_pert, label_pert_adv)

def adv_data_processor(df_adv, tokenizer, args, train=True):
    text_adv = list(df_adv['pert'].values)
    label_adv = df_adv['ground_truth_output'].values
    if train==True:
        label_adv = list(label_adv.astype(int)+args.num_classes)
    else:
        label_adv = list(label_adv.astype(int))

    out = tokenizer(text_adv)
    text_adv = tokenizer.batch_decode(out['input_ids'], skip_special_tokens=True)
    #data_adv = tokenizer(text_adv, padding=True, truncation=True)

    return text_adv, label_adv

def adv_det_data_processor(df_adv, tokenizer, args, train=True):

    # Original
    text_orig = list(df_adv['orig'].values) # Orig text
    out = tokenizer(text_orig)
    text_orig = tokenizer.batch_decode(out['input_ids'], skip_special_tokens=True)

    label_orig = df_adv['ground_truth_output'].values # Orig text label
    label_orig = list(label_orig.astype(int))

    # AdvExample 
    df_sc_adv = df_adv[df_adv['result_type']=='Successful'].reset_index()
    text_adv_pert = list(df_sc_adv['pert'].values)
    out = tokenizer(text_adv_pert)
    text_adv_pert = tokenizer.batch_decode(out['input_ids'], skip_special_tokens=True)

    label_pert_adv = df_sc_adv['ground_truth_output'].values
    label_pert_adv = list(label_pert_adv.astype(int))

    return (text_orig, label_orig), (text_adv_pert, label_pert_adv)


def trans_aug_dataloader3(dataset, df_adv_train_1, df_adv_train_2, tokenizer, ratio, args):
    if dataset == 'ag':
        """
        0: world
        1: sports
        2: buisiness
        3: Sci/Tech
        """
        dataset = load_dataset("ag_news")
        test, train = dataset['test'], dataset['train']

    elif dataset == 'mr':
        """
        Sentiment polarity datasets: binary 0/1
        """
        dataset = load_dataset("rotten_tomatoes")
        test, train = dataset['test'], dataset['train']

    elif dataset == 'imdb':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        dataset = load_dataset("imdb")
        test, train = dataset['test'], dataset['train']

    train_adv_text, train_adv_label = adv_data_processor2(df_adv_train_1, tokenizer, args, train=True)
    #train_adv_text2, train_adv_label2 = adv_data_processor2(df_adv_train_2, tokenizer, args, train=True)
    #train_adv_label2 = [x+args.num_classes for x in train_adv_label2]

    print(f"Adv1: {len(train_adv_text)}") 
    #print(f"Adv2: {len(train_adv_text2)}") 

    #test_adv_text, test_adv_label = adv_data_processor(df_adv_test, tokenizer, args, train=False)

    train_txt = train['text']+train_adv_text
    #train_txt = train['text']+train_adv_text+train_adv_text2
    #test_txt = test_adv_text
    test_txt = test['text']
    #test_txt = test['text']+test_adv_text

    print(f"Trainset Size: {len(train_txt)}")
    print(f"Testset Size: {len(test_txt)}")

    train_label = train['label']+train_adv_label
    #train_label = train['label']+train_adv_label+train_adv_label2
    #test_label = test_adv_label
    test_label = test['label']
    #test_label = test['label']+test_adv_label

#        train_data = tokenizer(train_txt, padding=True, truncation=True)
#        test_data = tokenizer(test_txt, padding=True, truncation=True)

    v_builder = Vocab_builder(train['text'], tokenizer)
    train_data, test_data = v_builder.build_masked_dataset(train_txt, test_txt, ratio, tokenizer)

    train_set = ClsDataset(train_data, train_label)
    test_set = ClsDataset(test_data, test_label)

#        if args.min_length_filter == True:
#            train_set = min_length_filter(train_set, sample_mask_train)
#            test_set = min_length_filter(test_set, sample_mask_test)
#            print(f"Subset Trainset Size: {len(train_set)}")
#            print(f"Subset Testset Size: {len(test_set)}")

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader

class ClsDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, ood_label):
        self.encodings = encodings
        self.labels = labels
        self.ood_label = ood_label

    def __getitem__(self, idx):
        
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['idx'] = idx
        item['ood_label'] = torch.tensor(self.ood_label[idx])

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

def sample_mask_fn(dataset, min_length=3):
    text = dataset['text']
    sampled_mask = list(range(len(text)))
    for i, txt in enumerate(text):
        length = len(txt.split(" "))
        if length < min_length:
            sampled_mask.remove(i)

    return sampled_mask

def sample_mask_fn2(text_dataset, min_length=3):
    text = text_dataset
    sampled_mask = list(range(len(text)))
    for i, txt in enumerate(text):
        length = len(txt.split(" "))
        if length < min_length:
            sampled_mask.remove(i)

    return sampled_mask

def min_length_filter(dataset, sample_mask):
    subset = torch.utils.data.Subset(dataset, sample_mask)
    return subset
    

    #length = sum([1 for i in val[idx] if i!=0])

def collate_fn(batch):

#    input_ids = batch['input_ids']
#    attention_mask = batch['attention_mask']
#    labels = batch['labels']

    ids_list = []
    attn_list = []
    label_list = []
    for data in batch:
        input_ids = data['input_ids']
        length = sum([1 for i in input_ids if i!=0])

#    for ids, attn, label in zip(input_ids, attention_mask, labels):
#        ids_list
#        label_list.append(torch.tensor(label))
#
#    padded_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=args.pad_idx)
#
#    padded_prem = torch.nn.utils.rnn.pad_sequence(prem_ids_list, batch_first=True, padding_value=args.pad_idx)
#
#    seq_len = torch.tensor(len_list)
#    prem_len = torch.tensor(prem_len_list)
#
#    labels = torch.tensor(labels)
#
#    padded_order = torch.nn.utils.rnn.pad_sequence(ord_list, batch_first=True, padding_value=1000)

    return batch

def lstm_dataloader(dataset, tokenizer, ratio, args):

    if dataset == 'ag':
        """
        0: world
        1: sports
        2: buisiness
        3: Sci/Tech
        """
        dataset = load_dataset("ag_news")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split = train.train_test_split(test_size=0.3)
            train = split['test']  # We want to 30% of the original trainset, so take 'test'

        if args.prep:
            test = test.map(preprocess3)
            train = train.map(preprocess3)

    elif 'clinc' in dataset:
        if dataset=='clinc':
            source_path = "./data/clinic/clinic150.json"
        elif dataset == 'clinc_ood':
            source_path = "./data/clinic/data_imbalanced.json"
        elif dataset == 'clinc_oop':
            source_path = "./data/clinic/data_oos_plus.json"
        elif dataset == 'clinc_small':
            source_path = "./data/clinic/data_small.json"
        #source_path = os.path.join(self.root_dir, '50EleReviews.json')
        with open(source_path, encoding='utf-8') as f:
            data = json.load(f)

        train_data = np.array(data['train'])
        test_data = np.array(data['test']) 
        train_ood_data = np.array(data['oos_train'])
        test_ood_data  = np.array(data['oos_test'])

        train_txt = list(train_data[:,0])
        train_label_ = list(train_data[:,1])

        label_set = np.unique(train_label_) # returns the "sorted" unique elements
        label_dict = {label:i for i, label in enumerate(label_set)}

        def txt2label(labels, label_dict):
            return [label_dict[l] for l in labels]

        train_label = txt2label(train_label_, label_dict)

        test_txt = list(test_data[:,0])
        test_label_ = list(test_data[:,1])
        test_label = txt2label(test_label_, label_dict)

        train_ood_txt = list(train_ood_data[:,0])
        #train_ood_label = list(train_ood_data[:,1])
        train_ood_label = [0 for i in range(len(train_ood_txt))]

        test_ood_txt = list(test_ood_data[:,0])
        #test_ood_label = list(test_ood_data[:,1])
        test_ood_label = [0 for i in range(len(test_ood_txt))]

        print(f"TrainIDset Size: {len(train_txt)} || TestIDset Size: {len(test_txt)}")
        print(f"TrainOODset Size: {len(train_ood_txt)} || TestOODset Size: {len(test_ood_txt)}")
        args.num_classes = np.unique(train_label).shape[0]

        train_id_data = tokenizer(train_txt, padding=True, truncation=True)
        test_id_data = tokenizer(test_txt, padding=True, truncation=True)

        train_ood_data = tokenizer(train_ood_txt, padding=True, truncation=True)
        test_ood_data = tokenizer(test_ood_txt, padding=True, truncation=True)

        train_id_set = ClsDataset(train_id_data, train_label)
        test_id_set = ClsDataset(test_id_data, test_label)

        train_ood_set = ClsDataset(train_ood_data, train_ood_label)
        test_ood_set = ClsDataset(test_ood_data, test_ood_label)

        train_id_dataloader = DataLoader(train_id_set, batch_size=args.batch_size, shuffle=True)
        test_id_dataloader = DataLoader(test_id_set, batch_size=args.batch_size, shuffle=True)
        train_ood_dataloader = DataLoader(train_ood_set, batch_size=args.batch_size, shuffle=True)
        test_ood_dataloader = DataLoader(test_ood_set, batch_size=args.batch_size, shuffle=True)

        return train_id_dataloader, test_id_dataloader, train_ood_dataloader, test_ood_dataloader

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
        args.n_vocab = tokenizer.vocab_size

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

        if args.split == True:
            split_train = train.train_test_split(test_size=0.4)
            split_test = train.train_test_split(test_size=args.split_p)
            train = split_train['test']  # We want to 30% of the original trainset, so take 'test'
            test = split_test['test']  # We want to 30% of the original trainset, so take 'test'

#        if args.prep:
#            test = test.map(preprocess_html)
#            train = train.map(preprocess_html)

    else:
        print("Specift dataset correctly")
        exit(0)

    print(f"Trainset Size: {len(train)}")
    print(f"Testset Size: {len(test)}")

    train_data = tokenizer(train['text'], padding=True, truncation=True)
    test_data = tokenizer(test['text'], padding=True, truncation=True)

#    v_builder = Vocab_builder(train['text'])
#    train_data, test_data = v_builder.build_dataset(train['text'], test['text'], ratio, tokenizer)

    args.n_vocab = tokenizer.vocab_size
    #args.n_vocab = len(v_builder.word2idx)
    #print(f"N_Vocab: {args.n_vocab}") 
    
    # dev_data = tokenizer(dev['text'], padding=True, truncation=True)

    train_label = train['label']
    test_label = test['label']
    # dev_label = dev['label']

    train_set = ClsDataset(train_data, train_label)
    test_set = ClsDataset(test_data, test_label)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader
    #return train_dataloader, test_dataloader, v_builder


def trans_aug_dataloader(dataset, df_adv_train, tokenizer, ratio, args):
    if dataset == 'ag':
        """
        0: world
        1: sports
        2: buisiness
        3: Sci/Tech
        """
        dataset = load_dataset("ag_news")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split = train.train_test_split(test_size=0.3)
            train = split['test']  # We want to 30% of the original trainset, so take 'test'

        print(f"Trainset Size: {len(train)}")
        print(f"Testset Size: {len(test)}")

        if args.prep:
            test = test.map(preprocess3)
            train = train.map(preprocess3)

        train_vocab = unk_tokenizer_setup(train['text'], args.p_vocab)

        train_mask = copy.deepcopy(train['text'])
        test_mask = copy.deepcopy(test['text'])

        train_mask = dataset_mask(train_mask, train_vocab, tokenizer)
        test_mask = dataset_mask(test_mask, train_vocab, tokenizer)

        train_data = tokenizer(train_mask, padding=True, truncation=True)
        test_data = tokenizer(test_mask, padding=True, truncation=True)

        train_label = train['label']
        test_label = test['label']

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

        train_adv_text, train_adv_label = adv_data_processor(df_adv_train, tokenizer, args, train=False)

        train_txt = train['text']+train_adv_text
        #test_txt = test_adv_text
        test_txt = test['text']
        #test_txt = test['text']+test_adv_text

        if args.min_length_filter == True:
            sample_mask_train = sample_mask_fn2(train_txt, min_length=args.min_length)
            sample_mask_test = sample_mask_fn2(test_txt, min_length=args.min_length)

        print(f"Trainset Size: {len(train_txt)}")
        print(f"Testset Size: {len(test_txt)}")

        #test_label = test_adv_label
        test_label = test['label']
        #test_label = test['label']+test_adv_label

#        train_data = tokenizer(train_txt, padding=True, truncation=True)
#        test_data = tokenizer(test_txt, padding=True, truncation=True)

        v_builder = Vocab_builder(train['text'], tokenizer)
#        train_data, test_data = v_builder.build_masked_dataset(train_txt, test_txt, ratio, tokenizer)
        train_data, test_data = v_builder.build_aug_masked_dataset(train_txt, test_txt, ratio, tokenizer)

        train_label = train['label']+train_adv_label
        train_label = train_label*2
        test_label = test['label']*2

        train_set = ClsDataset(train_data, train_label)
        test_set = ClsDataset(test_data, test_label)

        if args.min_length_filter == True:
            train_set = min_length_filter(train_set, sample_mask_train)
            test_set = min_length_filter(test_set, sample_mask_test)
            print(f"Subset Trainset Size: {len(train_set)}")
            print(f"Subset Testset Size: {len(test_set)}")

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    elif dataset == 'imdb':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        dataset = load_dataset("imdb")
        test, train = dataset['test'], dataset['train']

        train_adv_text, train_adv_label = adv_data_processor(df_adv_train, tokenizer, args, train=True)
        test_adv_text, test_adv_label = adv_data_processor(df_adv_test, tokenizer, args, train=False)

        train_txt = train['text']+train_adv_text
        #test_txt = test_adv_text
        test_txt = test['text']
        #test_txt = test['text']+test_adv_text

        print(f"Trainset Size: {len(train_txt)}")
        print(f"Testset Size: {len(test_txt)}")

        train_label = train['label']+train_adv_label
        #test_label = test_adv_label
        test_label = test['label']
        #test_label = test['label']+test_adv_label

#        train_data = tokenizer(train_txt, padding=True, truncation=True)
#        test_data = tokenizer(test_txt, padding=True, truncation=True)

        v_builder = Vocab_builder(train['text'], tokenizer)
        train_data, test_data = v_builder.build_masked_dataset(train_txt, test_txt, ratio, tokenizer)

        train_set = ClsDataset(train_data, train_label)
        test_set = ClsDataset(test_data, test_label)

#        if args.min_length_filter == True:
#            train_set = min_length_filter(train_set, sample_mask_train)
#            test_set = min_length_filter(test_set, sample_mask_test)
#            print(f"Subset Trainset Size: {len(train_set)}")
#            print(f"Subset Testset Size: {len(test_set)}")

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, test_dataloader

def mask_aug_dataloader(dataset, tokenizer, ratio, args):
    if dataset == 'ag':
        """
        0: world
        1: sports
        2: buisiness
        3: Sci/Tech
        """
        dataset = load_dataset("ag_news")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split = train.train_test_split(test_size=0.3)
            train = split['test']  # We want to 30% of the original trainset, so take 'test'

        print(f"Trainset Size: {len(train)}")
        print(f"Testset Size: {len(test)}")

        if args.prep:
            test = test.map(preprocess3)
            train = train.map(preprocess3)

        v_builder = Vocab_builder(train['text'], tokenizer)
        train_data, test_data = v_builder.build_masked_dataset(train['text'], test['text'], ratio, tokenizer)

        train_label = train['label']
        test_label = test['label']

#        train_data = tokenizer(train_mask, padding=True, truncation=True)
#        test_data = tokenizer(test_mask, padding=True, truncation=True)
#
#        train_label = train['label']
#        test_label = test['label']

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

        if args.min_length_filter == True:
            sample_mask_test = sample_mask_fn(test, min_length=args.min_length)
            sample_mask_train = sample_mask_fn(train)

        print(f"Trainset Size: {len(train)}")
        print(f"Testset Size: {len(test)}")

        if args.prep:
            test = test.map(preprocess)
            train = train.map(preprocess)

        v_builder = Vocab_builder(train['text'], tokenizer)
        train_data, test_data = v_builder.build_aug_masked_dataset(train['text'], test['text'], ratio, tokenizer)

        mask_label_train = list(np.array(train['label'])+args.num_classes)
        mask_label_test = list(np.array(test['label'])+args.num_classes)

        train_label = train['label']+mask_label_train
        test_label = test['label']+mask_label_test

#        train_label = train['label']*2
#        test_label = test['label']*2
        # dev_label = dev['label']

        train_set = ClsDataset(train_data, train_label)
        test_set = ClsDataset(test_data, test_label)

        if args.min_length_filter == True:
            train_set = min_length_filter(train_set, sample_mask_train)
            test_set = min_length_filter(test_set, sample_mask_test)
            print(f"Subset Trainset Size: {len(train_set)}")
            print(f"Subset Testset Size: {len(test_set)}")

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    elif dataset == 'imdb':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        dataset = load_dataset("imdb")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split_train = train.train_test_split(test_size=0.4)
            split_test = train.train_test_split(test_size=args.split_p)
            train = split_train['test']  # We want to 30% of the original trainset, so take 'test'
            test = split_test['test']  # We want to 30% of the original trainset, so take 'test'

        print(f"Trainset Size: {len(train)}")
        print(f"Testset Size: {len(test)}")

        if args.prep:
            test = test.map(preprocess_html)
            train = train.map(preprocess_html)

        v_builder = Vocab_builder(train['text'], tokenizer)
        train_data, test_data = v_builder.build_aug_masked_dataset(train['text'], test['text'], ratio, tokenizer)

#        mask_label_train = list(np.array(train['label']))
#        mask_label_test = list(np.array(test['label']))

        train_label = train['label']+train['label']
        test_label = test['label']+test['label']

        train_set = ClsDataset(train_data, train_label)
        test_set = ClsDataset(test_data, test_label)

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, test_dataloader

def text_dataloader_ood(dataset, args):

    if dataset == 'mnli':
        dataset = load_dataset("multi_nli")
        train, test_id, test_ood = dataset['train'], dataset['validation_matched'], dataset['validation_mismatched']

        if args.split == True:
            split = train.train_test_split(test_size=0.3)
            train = split['test'] # We want to 30% of the original trainset, so take 'test'

        print(f"Trainset Size: {len(train)}") 
        print(f"Testset ID Size: {len(test_id)}") 
        print(f"Testset OOD Size: {len(test_ood)}") 

        train_label = train['label']
        test_id_label = test_id['label']
        test_ood_label = test_ood['label']
        args.num_classes = 3

    elif dataset == 'imdb':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        args.num_classes = 2
        dataset = load_dataset("imdb")
        test = dataset['test']

    elif dataset == 'yelp':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        args.num_classes = 2
        dataset = load_dataset("yelp_polarity")
        test = dataset['test']

    else:
        print("Specift dataset correctly")
        exit(0)

    print(f"Testset Size: {len(test)}")


    return test


def spacy_tokenize(txt_dataset):
    import spacy
    from spacy.lang.en import English

    nlp = English()
    spacy_tokenizer = nlp.tokenizer

    txt_list = []
    for txt in txt_dataset:
        clean = clean_str(txt, tokenizer=spacy_tokenizer)
        txt_list.append(" ".join(clean))
    return txt_list

def text_dataloader_spacy(dataset, args):

    if dataset == 'ag':
        """
        0: world
        1: sports
        2: buisiness
        3: Sci/Tech
        """
        dataset = load_dataset("ag_news")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split = train.train_test_split(test_size=0.3)
            train = split['test']  # We want to 30% of the original trainset, so take 'test'

        if args.prep:
            test = test.map(preprocess3)
            train = train.map(preprocess3)
        args.num_classes = 4

    elif dataset == 'mr':
        """
        Sentiment polarity datasets: binary 0/1
        """
        dataset = load_dataset("rotten_tomatoes")
        test, train = dataset['test'], dataset['train']

#        if args.min_length_filter == True:
#            sample_mask_test = sample_mask_fn(test, min_length=args.min_length)
#            sample_mask_train = sample_mask_fn(train)
#
#        print(f"Trainset Size: {len(train)}")
#        print(f"Testset Size: {len(test)}")
#
#        if args.prep:
#            test = test.map(preprocess)
#            train = train.map(preprocess)

        args.num_classes = 2
#        train_label = train['label']
#        test_label = test['label']
#        # dev_label = dev['label']
#
#        return train_dataloader, test_dataloader

#    elif dataset == 'mnli':
#        dataset = load_dataset("multi_nli")
#        train, test_id, test_ood = dataset['train'], dataset['validation_matched'], dataset['validation_mismatched']
#
#        if args.split == True:
#            split = train.train_test_split(test_size=0.3)
#            train = split['test'] # We want to 30% of the original trainset, so take 'test'
#
#        print(f"Trainset Size: {len(train)}") 
#        print(f"Testset ID Size: {len(test_id)}") 
#        print(f"Testset OOD Size: {len(test_ood)}") 
#
#        train_list = []
#        for prem, hypo, label in zip(train['premise'], train['hypothesis'], train['label']):
#            tup = ((prem, hypo), label)
#            train_list.append(tup)
#            pdb.set_trace() 
#
#
#        dataset = textattack.datasets.Dataset(train_list, input_columns=("premise", "hypothesis"))
##        train_data = tokenizer(, padding=True, truncation=True)
##        test_id_data = tokenizer(test_id['premise'], test_id['hypothesis'], padding=True, truncation=True)
##        test_ood_data = tokenizer(test_ood['premise'], test_ood['hypothesis'], padding=True, truncation=True)
#
#        test_id_label = test_id['label']
#        test_ood_label = test_ood['label']
#        args.num_classes = 3
#
#        train_dict = {'text': train_txt, 'label': train_label}
#        train = Dataset.from_dict(train_dict)
#
#        test_dict = {'text': test_txt, 'label': test_label}
#        test = Dataset.from_dict(test_dict)
#
#
#        pdb.set_trace() 
#
#        return train_dataloader, test_id_dataloader, test_ood_dataloader

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

        train_txt = spacy_tokenize(train_txt)

        test_label = test['label']
        test_label = np.array(test_label)
        pos_idx = test_label>0.6
        neg_idx = test_label<0.4

        pos_label = np.ceil(test_label[pos_idx]).astype(int)
        neg_label = np.floor(test_label[neg_idx]).astype(int)

        test_txt = np.array(spacy_tokenize(test['sentence']))
        pos_txt = test_txt[pos_idx]
        neg_txt = test_txt[neg_idx]

        # Test
        test_txt = list(pos_txt)+list(neg_txt)
        test_label = list(pos_label)+list(neg_label)
        test_label = [int(i) for i in test_label]

        train_dict = {'text': train_txt, 'label': train_label}
        train = Dataset.from_dict(train_dict)

        test_dict = {'text': test_txt, 'label': test_label}
        test = Dataset.from_dict(test_dict)

        return train, test

    elif dataset == 'imdb':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        args.num_classes = 2
        dataset = load_dataset("imdb")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split_train = train.train_test_split(test_size=0.4)
            split_test = train.train_test_split(test_size=args.split_p)
            train = split_train['test']  # We want to 30% of the original trainset, so take 'test'
            test = split_test['test']  # We want to 30% of the original trainset, so take 'test'

    elif dataset == 'yelp':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        args.num_classes = 2
        dataset = load_dataset("yelp_polarity")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split_train = train.train_test_split(test_size=0.4)
            split_test = train.train_test_split(test_size=args.split_p)
            train = split_train['test']  # We want to 30% of the original trainset, so take 'test'
            test = split_test['test']  # We want to 30% of the original trainset, so take 'test'

    else:
        print("Specift dataset correctly")
        exit(0)

    print(f"Trainset Size: {len(train)}")
    print(f"Testset Size: {len(test)}")

    train_txt = spacy_tokenize(train['text'])
    test_txt = spacy_tokenize(test['text'])

    train_label = train['label']
    test_label = test['label']

    train_dict = {'text': train_txt, 'label': train_label}
    train = Dataset.from_dict(train_dict)

    test_dict = {'text': test_txt, 'label': test_label}
    test = Dataset.from_dict(test_dict)

    return train, test

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

        if args.split == True:
            split = train.train_test_split(test_size=0.3)
            train = split['test']  # We want to 30% of the original trainset, so take 'test'

        if args.prep:
            test = test.map(preprocess3)
            train = train.map(preprocess3)
        args.num_classes = 4

    elif dataset == 'mr':
        """
        Sentiment polarity datasets: binary 0/1
        """
        dataset = load_dataset("rotten_tomatoes")
        test, train = dataset['test'], dataset['train']

#        if args.min_length_filter == True:
#            sample_mask_test = sample_mask_fn(test, min_length=args.min_length)
#            sample_mask_train = sample_mask_fn(train)
#
#        print(f"Trainset Size: {len(train)}")
#        print(f"Testset Size: {len(test)}")
#
#        if args.prep:
#            test = test.map(preprocess)
#            train = train.map(preprocess)

        args.num_classes = 2
#        train_label = train['label']
#        test_label = test['label']
#        # dev_label = dev['label']
#
#        return train_dataloader, test_dataloader

    elif dataset == 'mnli':
        dataset = load_dataset("multi_nli")
        train, test_id, test_ood = dataset['train'], dataset['validation_matched'], dataset['validation_mismatched']

        if args.split == True:
            split = train.train_test_split(test_size=0.3)
            train = split['test'] # We want to 30% of the original trainset, so take 'test'

        print(f"Trainset Size: {len(train)}") 
        print(f"Testset ID Size: {len(test_id)}") 
        print(f"Testset OOD Size: {len(test_ood)}") 

        train_label = train['label']
        test_id_label = test_id['label']
        test_ood_label = test_ood['label']
        args.num_classes = 2

#        return train_dataloader, test_id_dataloader, test_ood_dataloader

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

        if args.split == True:
            split_train = train.train_test_split(test_size=0.4)
            split_test = train.train_test_split(test_size=args.split_p)
            train = split_train['test']  # We want to 30% of the original trainset, so take 'test'
            test = split_test['test']  # We want to 30% of the original trainset, so take 'test'

    else:
        print("Specift dataset correctly")
        exit(0)

    print(f"Trainset Size: {len(train)}")
    print(f"Testset Size: {len(test)}")


    return train, test

def adv_train_dataloader(dataset, df_adv, tokenizer, args):
    """
    - train with adversarial examples and clean samples
    """
    a_text = df_adv['pert']
    c_text = df_adv['orig']
    y = df_adv['ground_truth_output']

    text_orig = list(c_text.values) # Orig text
    text_pert = list(a_text.values) # Orig text
    #out = tokenizer(text_orig)

    y_orig = y.values # Orig text label
    y_orig = list(y_orig.astype(int))

    if dataset == 'ag':
        """
        0: world
        1: sports
        2: buisiness
        3: Sci/Tech
        """
        dataset = load_dataset("ag_news")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split = train.train_test_split(test_size=0.3)
            train = split['test']  # We want to 30% of the original trainset, so take 'test'

        if args.prep:
            test = test.map(preprocess3)
            train = train.map(preprocess3)

    elif dataset == 'imdb':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        dataset = load_dataset("imdb")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split_train = train.train_test_split(test_size=0.4)
            split_test = train.train_test_split(test_size=args.split_p)
            train = split_train['test']  # We want to 30% of the original trainset, so take 'test'
            test = split_test['test']  # We want to 30% of the original trainset, so take 'test'

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

#        train_txt = train['text']+text_pert
#        test_txt = test['text']
#
#        train_data = tokenizer(train_txt, padding=True, truncation=True)
#        test_data = tokenizer(test['text'], padding=True, truncation=True)
#        # dev_data = tokenizer(dev['text'], padding=True, truncation=True)
#
#        train_label = train['label']+y_orig
#        test_label = test['label']

        train_data = tokenizer(train_txt+text_pert, padding=True, truncation=True)
        test_data = tokenizer(test_txt, padding=True, truncation=True)

        train_set = ClsDataset(train_data, train_label+y_orig)
        test_set = ClsDataset(test_data, test_label)

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    elif dataset == 'yelp':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        dataset = load_dataset("yelp_polarity")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split_train = train.train_test_split(test_size=0.4)
            split_test = train.train_test_split(test_size=args.split_p)
            train = split_train['test']  # We want to 30% of the original trainset, so take 'test'
            test = split_test['test']  # We want to 30% of the original trainset, so take 'test'

#        if args.prep:
#            test = test.map(preprocess_html)
#            train = train.map(preprocess_html)

    else:
        print("Specift dataset correctly")
        exit(0)

    print(f"Trainset Size: {len(train)}")
    print(f"Testset Size: {len(test)}")
    print(f"AdvEx dataset Size: {len(df_adv)}")

    train_txt = train['text']+text_pert
    test_txt = test['text']

    train_data = tokenizer(train_txt, padding=True, truncation=True)
    test_data = tokenizer(test['text'], padding=True, truncation=True)
    # dev_data = tokenizer(dev['text'], padding=True, truncation=True)

    train_label = train['label']+y_orig
    test_label = test['label']
    # dev_label = dev['label']

    train_set = ClsDataset(train_data, train_label)
    test_set = ClsDataset(test_data, test_label)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def adv_train_dataloader_v3(df_adv, tokenizer, args):
    """
    - train only with adversarial examples
    """

    a_text = df_adv['pert']
    c_text = df_adv['orig']
    y = df_adv['ground_truth_output']

    #n_query = df_adv['n_query']

    text_orig = list(c_text.values) # Orig text
    text_pert = list(a_text.values) # Orig text
    #out = tokenizer(text_orig)

    y_orig = y.values # Orig text label
    y_orig = list(y_orig.astype(int))

    print(f"AdvEx dataset Size: {len(text_pert+text_orig)}")

    test_data = tokenizer(text_pert, padding=True, truncation=True)
    #test_data = tokenizer(text_orig+text_pert, padding=True, truncation=True)

    #train_data = tokenizer(text_pert, padding=True, truncation=True)
    #dev_data = tokenizer(dev['text'], padding=True, truncation=True)

    #train_label = y_orig
    test_label = y_orig
    #test_label = y_orig+y_orig

    #train_set = ClsDataset(train_data, train_label)
    test_set = ClsDataset(test_data, test_label)

    #train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    return test_dataloader

def adv_train_dataloader_v2(df_adv, tokenizer, args):
    """
    - train only with adversarial examples
    """

    a_text = df_adv['pert']
    y_a = df_adv['pert_pred']

    c_text = df_adv['orig']
    y = df_adv['ground_truth_output']
    #n_query = df_adv['n_query']

    text_orig = list(c_text.values) # Orig text
    text_pert = list(a_text.values) # Orig text
    #out = tokenizer(text_orig)

    y_orig = y.values # Orig text label
    y_orig = list(y_orig.astype(int))

    print(f"AdvEx dataset Size: {len(text_pert)}")

    train_data = tokenizer(text_pert, padding=True, truncation=True)
    test_data = tokenizer(text_orig, padding=True, truncation=True)
    # dev_data = tokenizer(dev['text'], padding=True, truncation=True)

    train_label = y_orig
    test_label = y_orig

    train_set = ClsDataset(train_data, train_label)
    test_set = ClsDataset(test_data, test_label)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def rat_train_dataloader(dataset, df_adv, tokenizer, args):

    a_text = df_adv['pert']
    y_a = df_adv['pert_pred']

    c_text = df_adv['orig']
    y = df_adv['ground_truth_output']
    #n_query = df_adv['n_query']

    text_orig = list(c_text.values) # Orig text
    text_pert = list(a_text.values) # Orig text
    #out = tokenizer(text_orig)

    y_orig = y.values # Orig text label
    #y_orig = list(y_orig.astype(int))

    if args.dataset_type=='train':
        y_orig = list(y_orig.astype(int)+args.num_classes)
    else:
        y_orig = list(y_orig.astype(int))

    if dataset == 'ag':
        """
        0: world
        1: sports
        2: buisiness
        3: Sci/Tech
        """
        dataset = load_dataset("ag_news")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split = train.train_test_split(test_size=0.3)
            train = split['test']  # We want to 30% of the original trainset, so take 'test'

        if args.prep:
            test = test.map(preprocess3)
            train = train.map(preprocess3)

    elif dataset == 'imdb':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        dataset = load_dataset("imdb")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split_train = train.train_test_split(test_size=0.4)
            split_test = train.train_test_split(test_size=args.split_p)
            train = split_train['test']  # We want to 30% of the original trainset, so take 'test'
            test = split_test['test']  # We want to 30% of the original trainset, so take 'test'

    elif dataset == 'mr':
        """
        Sentiment polarity datasets: binary 0/1
        """
        dataset = load_dataset("rotten_tomatoes")
        test, train = dataset['test'], dataset['train']

        if args.min_length_filter == True:
            sample_mask_test = sample_mask_fn(test, min_length=args.min_length)
            sample_mask_train = sample_mask_fn(train)

        print(f"Trainset Size: {len(train)}")
        print(f"Testset Size: {len(test)}")

        if args.prep:
            test = test.map(preprocess)
            train = train.map(preprocess)

        train_data = tokenizer(train['text'], padding=True, truncation=True)
        test_data = tokenizer(test['text'], padding=True, truncation=True)
        # dev_data = tokenizer(dev['text'], padding=True, truncation=True)

        train_label = train['label']
        test_label = test['label']
        # dev_label = dev['label']

        train_set = ClsDataset(train_data, train_label)
        test_set = ClsDataset(test_data, test_label)

        if args.min_length_filter == True:
            train_set = min_length_filter(train_set, sample_mask_train)
            test_set = min_length_filter(test_set, sample_mask_test)
            print(f"Subset Trainset Size: {len(train_set)}")
            print(f"Subset Testset Size: {len(test_set)}")

        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    elif dataset == 'mnli':
        dataset = load_dataset("multi_nli")
        train, test_id, test_ood = dataset['train'], dataset['validation_matched'], dataset['validation_mismatched']

        if args.split == True:
            split = train.train_test_split(test_size=0.3)
            train = split['test'] # We want to 30% of the original trainset, so take 'test'

        print(f"Trainset Size: {len(train)}") 
        print(f"Testset ID Size: {len(test_id)}") 
        print(f"Testset OOD Size: {len(test_ood)}") 

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

    elif dataset == 'yelp':
        """
        Sentiment polarity datasets: binary 0:neg/1:pos
        """
        dataset = load_dataset("yelp_polarity")
        test, train = dataset['test'], dataset['train']

        if args.split == True:
            split_train = train.train_test_split(test_size=0.4)
            split_test = train.train_test_split(test_size=args.split_p)
            train = split_train['test']  # We want to 30% of the original trainset, so take 'test'
            test = split_test['test']  # We want to 30% of the original trainset, so take 'test'

#        if args.prep:
#            test = test.map(preprocess_html)
#            train = train.map(preprocess_html)

    else:
        print("Specift dataset correctly")
        exit(0)

    print(f"Trainset Size: {len(train)}")
    print(f"Testset Size: {len(test)}")
    print(f"AdvEx dataset Size: {len(df_adv)}")

    train_txt = train['text']+text_pert
    test_txt = test['text']

    train_data = tokenizer(train_txt, padding=True, truncation=True)
    test_data = tokenizer(test['text'], padding=True, truncation=True)
    # dev_data = tokenizer(dev['text'], padding=True, truncation=True)

    train_label = train['label']+y_orig
    test_label = test['label']
    # dev_label = dev['label']

    train_set = ClsDataset(train_data, train_label)
    test_set = ClsDataset(test_data, test_label)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader
