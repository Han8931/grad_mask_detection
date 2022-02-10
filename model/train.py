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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

import pdb

def trainer_full(model, mlm, batch, optimizer, optimizer_mlm, stop_ids, args):

    input_ids = batch['input_ids'].to(args.device)
    attention_mask = batch['attention_mask'].to(args.device)
    labels = batch['labels'].to(args.device)

    loss_mlm = mlm(input_ids)
    loss_mlm.backward()
    optimizer_mlm.step()
    optimizer_mlm.zero_grad()

    masked_ids, mask_pos = masking_fn(input_ids, stop_ids, args)
    neg_ids = mlm.gen_negative_input(masked_ids, mask_pos, top_k=args.top_k)
    
    output = model(input_ids, masked_ids, neg_ids, labels) 
    loss = output['loss']

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()
    optimizer.zero_grad()

    return loss

def trainer(model, batch, optimizer, args):
    optimizer.zero_grad()           

    input_ids = batch['input_ids'].to(args.device)
    attention_mask = batch['attention_mask'].to(args.device)
    labels = batch['labels'].to(args.device) 

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

    loss = outputs.loss
    loss.backward()    
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

    optimizer.step()   

    return loss.item()

def LinearScheduler(optimizer, total_iter, curr, lr_init):
    lr = -(lr_init / total_iter) * curr + lr_init
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def batch_len(input_ids, pad_idx=0):
    b_length = (input_ids != pad_idx).data.sum(dim=-1)
    return b_length


def mask_src(input_ids, batch_len):
    mask_len = batch_len
    r, c = input_ids.size(0), input_ids.size(1)
    # c = c*2-1
    src_mask = torch.zeros((r, c)).bool()
    for m_len, mask_ in zip(mask_len, src_mask):
        mask_[1:m_len - 1] = True
    return src_mask


def stopwords_id(tokenizer):
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                 "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'us', 'him', 'his', 'himself', 'she', "she's",
                 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                 'by', 'for', 'with', 'without', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', 'cannot', "needn't", 'shan', "shan't", 'shouldn',
                 "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'never', 'along',
                 'ing', 'er', 'ed', 'could', "[PAD]", "[SEP]", "[CLS]", '.', '!', ',', '?', '>', '<', '-', '@', '&', '/', '%', '-', '_', '+']

    #tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    stopwords_ = tokenizer(stopwords, padding=False)
    stopwords_ = stopwords_['input_ids']

    stopwords = [val for sublist in stopwords_ for val in sublist]
    stopwords = list(set(stopwords))
#    stopwords.remove(101)
#    stopwords.remove(102)

    return stopwords


def keyword_mask(input_ids, src_mask, stop_ids):
    key_mask = src_mask.clone()
    for i, inp_ids in enumerate(input_ids):
        for j, ids in enumerate(inp_ids):
            if ids in stop_ids:
                key_mask[i][j] = 0

    return key_mask


def WordNet(word: str):
    """
    WordNet is a large lexical database of English. 
    Nouns, verbs, adjectives and adverbs are grouped into sets of cognitive synonyms (synsets), 
    each expressing a distinct concept.

    - ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    """
    # assert type(word)=='str'

    syns = wordnet.synsets(word)

    synonyms = []
    for syn in syns:
        for l in syn.lemmas():
            syn_ = l.name()
            if syn_.lower() == word.lower():
                continue
            else:
                synonyms.append(syn_)
    #        if l.antonyms():
    #            antonyms.append(l.antonyms()[0].name())
    synonyms = list(set(synonyms))

    return synonyms


def get_synonyms_len(syn_dict: dict):
    length = len(syn_dict.get("syn_set"))
    if length==1:
        return length+1000
    else:
        return length


def search_synonym(input_ids, rand_mask, mask_idx, tokenizer):
    synonyms_batch = []
    for ids, mask_ in zip(input_ids, rand_mask):
        synonyms = []
        synonyms_size = []
        for m in mask_:
            word_ids = ids[m.item()] # original token idx
            word = tokenizer.decode(word_ids) # original token
            word_syns = WordNet(word) # synonyms of the original token
            word_syns.append(word)

            syn_dict = {"mask_idx": m.item(), "syn_set": word_syns}
            synonyms.append(syn_dict)

        synonyms.sort(key=get_synonyms_len)
        synonyms_batch.append(synonyms)

    return synonyms_batch


def read_syn_dict(syn_dict: dict):
    mask_idx = syn_dict['mask_idx']
    syn_set = syn_dict['syn_set']
    syn_len = len(syn_dict['syn_set'])
    return mask_idx, syn_set, syn_len

def random_masking(key_mask, seed=0):
    mask = key_mask.int()
    rand_mask = []
    perm_idx = []
    for i, mask_ in enumerate(mask):
        if mask_.sum()==0:
            mask_[1] = 1
            mask_i = (mask_ == 1).nonzero().squeeze(1)
        elif mask_.sum()==1: # Some inputs are all zero...
            mask_i = (mask_ == 1).nonzero()
        else:
            mask_i = (mask_ == 1).nonzero().squeeze()
        
        perm = torch.randperm(mask_i.size(0))
        rand_mask.append(mask_i)
        perm_idx.append(perm)

    return rand_mask, perm_idx


def rand_masking_input(input_ids, rand_mask, perm_idx, mask_p:str, mask_idx=0):
    """
    - Return masked input_ids: inp_ids
    """
    #assert type(mask_p)==float or type(mask_p)==int, "mask_p must be either int or float type"

    syn_set_list = []
    inp_ids = input_ids.clone()
    mask_pos = []

    # Masking for each sample in a batch
    for ids, r_mask, p_idx in zip(inp_ids, rand_mask, perm_idx):
        r_mask = r_mask[p_idx]

        if '.' not in mask_p:
            mask_p_ = int(mask_p)
        else:
            mask_p_ = float(mask_p)
            mask_p_ = int(len(r_mask)*mask_p_)

#        if mask_p_==0:
#            mask_p_ = 1

        m = r_mask[:mask_p_]
        mask_pos.append(m)
        
        ids[m] = mask_idx

    return inp_ids, mask_pos

def masking_fn(input_ids, stop_ids, args):
    """
    Return masked input_ids: masked_ids
    """
    b_length = batch_len(input_ids, args.pad_idx)
    src_mask = mask_src(input_ids, b_length).to(args.device)
    if args.rand_mask==True:
        rand_mask, perm_idx = random_masking(src_mask)
    else:
        key_mask = keyword_mask(input_ids, src_mask, stop_ids)
        rand_mask, perm_idx = random_masking(key_mask)

    masked_ids, mask_pos = rand_masking_input(input_ids, rand_mask, perm_idx, args.mask_p, mask_idx=args.mask_idx)
    return masked_ids, mask_pos


def masking_input(input_ids, synonym_set, tokenizer, nth_mask, mask_idx=0):
    """
    - Return masked input_ids
    - synonym_set: n_batch x n_masking
    """
    syn_set_list = []
    inp_ids = input_ids.clone()

    # Masking for each sample in a batch
    for ids, syns in zip(inp_ids, synonym_set):
        syn = syns[nth_mask]
        m, syn_set, syn_len = read_syn_dict(syn)
        syn_set.append(tokenizer.decode(ids[m]))
        
        ids[m] = mask_idx

        #torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False, padding_value=0.0)
        #syn_set_list.append(syn_set)
        syn_set_list.append(tokenizer(syn_set, padding=True, return_tensors="pt"))

    return inp_ids, syn_set_list

def masking_inputs(input_ids, synonym_set, tokenizer, args):
    with torch.no_grad():
        if args.num_mask > 1:
            masked_ids = []
            for n in range(num_mask):
                masked_inps, syn_set_list = masking_input(input_ids, synonym_set, tokenizer, nth_mask=n,
                                                          mask_idx=args.mask_idx)
                masked_ids.append(masked_inps)
            masked_ids = torch.cat(masked_ids, dim=0)
        else:
            masked_ids, syn_set_list = masking_input(input_ids, synonym_set, tokenizer, nth_mask=0,
                                                     mask_idx=args.mask_idx)

    return masked_ids, syn_set_list
