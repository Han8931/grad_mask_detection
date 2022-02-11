import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AdamW
#from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertForMaskedLM
from transformers import DistilBertForMaskedLM
from transformers import RobertaTokenizer, RobertaModel

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

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

from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

from model.train import LinearScheduler, masking_fn, batch_len
#from model.robust_train import onezero_encoder, max_loss

import math

from nltk.tokenize import word_tokenize


class Attention(nn.Module):
    def __init__(self, enc_dim):
        super(Attention, self).__init__()

        self.attn = nn.Linear(enc_dim + enc_dim, enc_dim)
        self.v = nn.Linear(enc_dim, 1, bias = False)

    def forward(self, sent_emb, enc_outs, src_mask, w_attn=0.5, ns_mask = None):
        """
        - At t=0, hidden is from Enc
        - After t=1, hidden is from Dec
        - hidden = [batch size, dec hid dim]
        - enc_outs = [src len, batch size, enc hid dim * 2]
        """
        b_length = src_mask.sum(dim=-1)
        src_len = enc_outs.size(1)

        # Repeat decoder hidden state src_len times--------------
        #hidden = [batch size, src len, dec hid dim]
        #enc_outs = [batch size, src len, enc hid dim]
        hidden = sent_emb.unsqueeze(1).repeat(1, src_len, 1)

        # Bahdanau attention
        # v x tanh(W_a s_{i-1}+U_a h_j)
        #energy = [batch size, src len, dec hid dim]
        energy = torch.tanh(self.attn(torch.cat((enc_outs, hidden), dim = 2)))
        attention = self.v(energy).squeeze(2)

        attention = attention.masked_fill(src_mask == 0, -1e4)
        if ns_mask is not None:
            attention[ns_mask] = attention[ns_mask]*w_attn
            #attention[ns_mask] = attention[ns_mask]
#        attention[:,0] = -1e4
#        attention[range(hidden.size(0)),b_length-1] = -1e4
        #attention = attention.masked_fill(src_mask == 0, -1e10)

        return attention.softmax(dim = -1)

class GradModel(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(GradModel, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device

        self.pad_idx = args.pad_idx
        self.mask_idx = args.mask_idx
        self.cls_token = args.cls_token
        self.sep_token = args.sep_token

        self.p_topk = int(args.p_prune*args.embed_dim) # Sparsity Rate
        self.multi_mask = args.multi_mask
        self.smooth_grad = args.smooth_grad
        self.n_smooth = args.n_smooth
        self.noise_eps = args.noise_eps

    def forward(self, input_ids, attention_mask, labels=None, delta_grad=None):
        """
        - x: input
        """
        if delta_grad is not None:
            abs_grad = torch.abs(delta_grad+0.000000001)
            val, indices_ = torch.topk(abs_grad, self.p_topk)
            pruned_emb = torch.scatter(abs_grad, 2, indices_, 0)
            prune_mask = pruned_emb!=0
            emb_x_, word_embeddings = self.emb_prune(input_ids, attention_mask, prune_mask=prune_mask)

        else:
            emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)
        logits = self.cls(emb_x) # output from a FC layer
        
        if labels is not None:

            loss_fn = nn.CrossEntropyLoss()
            loss_ce = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
            output = {'logits': logits, 'loss': loss_ce, 'emb': emb_x}

            return output

        else:
            output = {'logits': logits, 'emb': emb_x}

        return output

    def emb_prune(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None, 
        prune_mask=None,
        word_embeddings=None,):

        self.config = self.enc.encoder.config
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.enc.encoder.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.enc.encoder.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.enc.encoder.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.enc.encoder.get_head_mask(head_mask, self.config.num_hidden_layers)

        if word_embeddings==None:
            embedding_output = self.enc.encoder.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

            if self.smooth_grad:
                randn_noise = torch.randn_like(embedding_output)*self.noise_eps
                p_emb_out = (embedding_output+randn_noise)*prune_mask.float()

            else:
                p_emb_out = embedding_output*prune_mask.float()
        else:
            p_emb_out = word_embeddings

        encoder_outputs = self.enc.encoder.encoder(
            p_emb_out,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.enc.encoder.pooler(sequence_output) if self.enc.encoder.pooler is not None else None

        output =  BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,)

        return output, p_emb_out

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

        #if input_ids.shape[0]==1:

        if pred is None:
            #emb_x_, inputs_embeds = self.embedding_encoder(input_ids, attention_mask, return_emb=True)
            with torch.no_grad():
                out_dict = self.forward(input_ids, attention_mask)
            logits = out_dict['logits']
            pred = logits.softmax(dim=-1).argmax(dim=-1) # prediction label

        else:
            if input_ids.shape[0]==1:
                pred = torch.Tensor([pred]).long().to(self.device)

        delta_grad_ = self.get_emb_grad(input_ids, attention_mask, pred)
        delta_grad = delta_grad_[0].detach()

#        abs_grad = torch.abs(delta_grad)
#        #val, indices_ = torch.topk(abs_grad, topk)
#        val, indices_ = torch.topk(abs_grad, topk)
#
#        pruned_emb = torch.scatter(abs_grad, 2, indices_, 0)
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
            
        return indice_list, delta_grad

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

        output = self.forward(input_ids, attention_mask, labels=labels, delta_grad=None)
        loss = output['loss']

        loss.backward()

        # grad w.r.t to word embeddings
        #grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove() # Remove Hook

        return emb_grads

class SGradAdvTrWrapper():
    def __init__(self, model, args):
        self.model = model 

        self.num_classes = args.num_classes
        self.device = args.device

        self.pad_idx = args.pad_idx
        self.mask_idx = args.mask_idx
        self.cls_token = args.cls_token
        self.sep_token = args.sep_token

        self.p_topk = int(args.p_prune*args.embed_dim) # Sparsity Rate
        self.multi_mask = args.multi_mask
        self.smooth_grad = args.smooth_grad
        self.n_smooth = args.n_smooth
        self.noise_eps = args.noise_eps

    def inference(self, input_ids, attention_mask, labels=None):
        """
        - x: input
        """

        if labels==None:
            output = self.model(masked_ids, attention_mask)
            labels = output['logits'].argmax(dim=-1)

        embedding_list = []
        gradient_list = []

        for i in range(self.n_smooth):

            self.model.eval()
            indices, delta_grad = self.model.grad_detection_batch2(input_ids, attention_mask, topk=5, pred=labels, mask_filter=True)
            self.model.zero_grad()           
            self.model.train()

            # 2. Masking inputs
            masked_ids = input_ids.clone()
            for ids_, m_idx in zip(masked_ids, indices):
                for j in range(self.multi_mask):
                    ids_[m_idx[j]] = self.mask_idx

            if delta_grad is not None:
                abs_grad = torch.abs(delta_grad+0.000000001)
                val, indices_ = torch.topk(abs_grad, self.p_topk)
                pruned_emb = torch.scatter(abs_grad, 2, indices_, 0)
                prune_mask = pruned_emb!=0
                emb_x_, word_embeddings = self.model.emb_prune(input_ids, attention_mask, prune_mask=prune_mask)
            else:
                emb_x_ = self.model.enc.encoder(input_ids, attention_mask)


            embedding_list.append(word_embeddings)

        avg_embeddings = torch.stack(embedding_list, axis=0).mean(dim=0)
        emb_x_, word_embeddings = self.model.emb_prune(input_ids, attention_mask, prune_mask=prune_mask, word_embeddings = avg_embeddings)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)
        logits = self.model.cls(emb_x) # output from a FC layer

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss_ce = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
            output = {'logits': logits, 'loss': loss_ce, 'emb': emb_x}
            return output
        else:
            output = {'logits': logits, 'emb': emb_x}

        return output


class PGradAdvTr(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(PGradAdvTr, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device

        self.pad_idx = args.pad_idx
        self.mask_idx = args.mask_idx
        self.cls_token = args.cls_token
        self.sep_token = args.sep_token

        self.p_topk = int(args.p_prune*args.embed_dim) # Sparsity Rate
        self.multi_mask = args.multi_mask
        self.smooth_grad = args.smooth_grad
        self.n_smooth = args.n_smooth
        self.noise_eps = args.noise_eps

        #self.delta_grad = None
        #self.batchnorm = nn.BatchNorm1d(768, affine=False)

    def embeddings(self, input_ids, attention_mask):
        outputs = self.enc.encoder(input_ids, attention_mask, output_hidden_states=True)
        embs = outputs['hidden_states']
        emb_init = embs[0]
        emb_last = embs[-1]

        return emb_last, emb_init

#    @staticmethod
#    def forward_pre_hook(module, inputs):
#        a, b = inputs
#        return a+10, b
    
    def inference(self, input_ids, attention_mask, topk=5, labels=None, mask_filter=True):

        indices, delta_grad = self.grad_detection_batch2(input_ids, attention_mask, topk=5, pred=labels, mask_filter=True)
        #model.zero_grad()           

        masked_ids = input_ids.clone()
        for ids_, m_idx in zip(masked_ids, indices):
            for j in range(self.multi_mask):
                ids_[m_idx[j]] = self.mask_idx

        
        with torch.no_grad():

            abs_grad = torch.abs(delta_grad+0.000000001)
            val, indices_ = torch.topk(abs_grad, self.p_topk)
            pruned_emb = torch.scatter(abs_grad, 2, indices_, 0)
            prune_mask = pruned_emb!=0
            emb_x_ = self.emb_prune(input_ids, attention_mask, prune_mask=prune_mask)

            hidden_state = emb_x_[0]  # (bs, seq_len, dim)
            emb_x = hidden_state[:, 0]  # (bs, dim)
            logits = self.cls(emb_x) # output from a FC layer
            labels = logits.softmax(dim=-1).argmax(dim=-1) # prediction label

            loss_fn = nn.CrossEntropyLoss()
            loss_ce = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

            output = {'logits': logits, 'loss': loss_ce, 'emb': emb_x}

        return output

    def forward(self, input_ids, attention_mask, labels=None, delta_grad=None):
        """
        - x: input
        """
        #b_length = batch_len(input_ids, self.pad_idx)

        if delta_grad is not None:
            abs_grad = torch.abs(delta_grad+0.000000001)
            val, indices_ = torch.topk(abs_grad, self.p_topk)
            pruned_emb = torch.scatter(abs_grad, 2, indices_, 0)
            prune_mask = pruned_emb!=0
            emb_x_ = self.emb_prune(input_ids, attention_mask, prune_mask=prune_mask)

        else:
            emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)
        #emb_bn_x = self.batchnorm(emb_x)
        logits = self.cls(emb_x) # output from a FC layer
        

        if labels is not None:

            loss_fn = nn.CrossEntropyLoss()
            loss_ce = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

            output = {'logits': logits, 'loss': loss_ce, 'emb': emb_x}

            return output

        else:
            output = {'logits': logits, 'emb': emb_x}

        return output

    def emb_prune(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None, 
        prune_mask=None,):

        self.config = self.enc.encoder.config
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.enc.encoder.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.enc.encoder.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.enc.encoder.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if self.smooth_grad:
            randn_noise = torch.randn_like(embedding_output)*self.noise_eps
            p_emb_out = (embedding_output+randn_noise)*prune_mask.float()

        else:
            p_emb_out = embedding_output*prune_mask.float()


        encoder_outputs = self.enc.encoder.encoder(
            p_emb_out,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.enc.encoder.pooler(sequence_output) if self.enc.encoder.pooler is not None else None

        output =  BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,)

        return output

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

        #if input_ids.shape[0]==1:

        if pred is None:
            #emb_x_, inputs_embeds = self.embedding_encoder(input_ids, attention_mask, return_emb=True)
            with torch.no_grad():
                out_dict = self.forward(input_ids, attention_mask)
            logits = out_dict['logits']
            pred = logits.softmax(dim=-1).argmax(dim=-1) # prediction label

        else:
            if input_ids.shape[0]==1:
                pred = torch.Tensor([pred]).long().to(self.device)

        delta_grad = self.get_emb_grad(input_ids, attention_mask, pred)[0].detach()

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
            
        return indice_list, delta_grad

#    def register_hooks(self):
#    """Register forward and backward hook to Conv module."""
#        self.forward_hooks, self.backward_hooks= [], []
#        for module, name in self.conv_names.items():
#            self.forward_hooks.append(module.register_forward_hook(self.save_input_forward_hook))
#            self.backward_hooks.append(module.register_full_backward_hook(self.compute_fisher_backward_hook)
#
#    def remove_hooks(self):
#        for fh in self.forward_hooks:
#            fh.remove()
#        for bh in self.backward_hooks:
#            bh.remove()

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

    def random_indices_batch(self, input_ids, attention_mask, topk=5, pred=None):
        """
        - x: input
        """
        b_length = batch_len(input_ids, self.pad_idx)

        indice_list = []
        for b_len in b_length:
            temp = np.random.randint(1, b_len.item()-1, size=(topk+1))
            indice_list.append(temp)

        indice_list = np.array(indice_list)
        return indice_list

    

    def iterative_grad_mask_detection_batch(self, input_ids, attention_mask, pred_org, topk=10, indices=None, multi_mask=0, iterative=False):
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
            smp = logits.softmax(dim=-1) # Softmax Prob

            for i, p in enumerate(pred):
                conf = smp[i, pred_org[i].item()]
                conf_l.append(conf.item())
                pred_l.append(p.item())

            confidence_list.append(conf_l)
            prediction_list.append(pred_l)

            if iterative:
                indices_it = self.grad_detection_batch(masked_ids, attention_mask, 3, pred_org)

                for i, _ in enumerate(masked_ids):
                    try:
                        masked_ids[i][indices_it[i][0]] = self.mask_idx
                    except:
                        continue

            else:
                masked_ids = input_ids.clone()
                for k, _ in enumerate(masked_ids):
                    try:
                        masked_ids[k][indices[k][m_idx+1]] = self.mask_idx
                    except:
                        print("Skipped")
                        continue

        output = {'prediction_list': prediction_list, 'indices': indices, 'confidence': np.array(confidence_list)}

        return output


class GradAdvTr(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(GradAdvTr, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device

        self.pad_idx = args.pad_idx
        self.mask_idx = args.mask_idx
        self.cls_token = args.cls_token
        self.sep_token = args.sep_token
        self.multi_mask = args.multi_mask
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

        #if input_ids.shape[0]==1:

        if pred is None:
            #emb_x_, inputs_embeds = self.embedding_encoder(input_ids, attention_mask, return_emb=True)
            with torch.no_grad():
                out_dict = self.forward(input_ids, attention_mask)
            logits = out_dict['logits']
            pred = logits.softmax(dim=-1).argmax(dim=-1) # prediction label

        else:
            if input_ids.shape[0]==1:
                pred = torch.Tensor([pred]).long().to(self.device)

        delta_grad_ = self.get_emb_grad(input_ids, attention_mask, pred)
        delta_grad = delta_grad_[0].detach()

#        abs_grad = torch.abs(delta_grad)
#        #val, indices_ = torch.topk(abs_grad, topk)
#        val, indices_ = torch.topk(abs_grad, topk)
#
#        pruned_emb = torch.scatter(abs_grad, 2, indices_, 0)
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
            
        return indice_list

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
    

class ClsTextDet(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(ClsTextDet, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device
        self.pad_idx = args.pad_idx
        self.mask_idx = args.mask_idx

    def embeddings(self, input_ids, attention_mask):
        outputs = self.enc.encoder(input_ids, attention_mask, output_hidden_states=True)
        embs = outputs['hidden_states']
        emb_init = embs[0]
        emb_last = embs[-1]

        return emb_last, emb_init

    def random_indices_batch(self, input_ids, attention_mask, topk=5, pred=None):
        """
        - x: input
        """
        b_length = batch_len(input_ids, self.pad_idx)

        indice_list = []
        for b_len in b_length:
            temp = np.random.randint(1, b_len.item()-1, size=(topk+1))
            indice_list.append(temp)

        indice_list = np.array(indice_list)
        return indice_list

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

class ClsFreq(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(ClsFreq, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device
        self.pad_idx = args.pad_idx
        self.mask_idx = args.mask_idx

    def embeddings(self, input_ids, attention_mask):
        outputs = self.enc.encoder(input_ids, attention_mask, output_hidden_states=True)
        embs = outputs['hidden_states']
#        emb_init = embs[0]
#        emb_last = embs[-1]

        return embs

    def noisy_embedding_encoder(self, inputs_embeds, attention_mask):
        output_attentions = self.enc.encoder.config.output_attentions
        output_hidden_states = (self.enc.encoder.config.output_hidden_states)
        return_dict = self.enc.encoder.config.use_return_dict
        head_mask = self.enc.encoder.get_head_mask(None, self.enc.encoder.config.num_hidden_layers)

#        inputs_embeds = self.enc.encoder.embeddings(input_ids)
#        randn_noise = torch.randn_like(inputs_embeds)*self.epsilon
#        indices = torch.randperm(inputs_embeds.size(1))[:int(input_ids.size(1)*0.7)]
#        randn_noise[:,indices,:] = 0
#        inputs_embeds_noise = inputs_embeds+randn_noise

        output = self.enc.encoder.transformer(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,)
        return output

    def forward(self, input_ids, attention_mask, labels=None):
        """
        - x: input
        """
#        embeddings = self.embeddings(input_ids, attention_mask)
#        cls_emb_list = []
#        for i in range(len(embeddings)):
#            cls_emb_list.append(embeddings[i][:,0,:])
#
#        cls_emb = torch.stack(cls_emb_list, 1)
#        fourier_feature = torch.fft.fft(cls_emb)
#        fourier_feature_shifted = torch.fft.fftshift(fourier_feature)
        

        emb_x_ = self.enc.encoder(input_ids, attention_mask)

#        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
#        emb_x = hidden_state[:, 0]  # (bs, dim)

        emb_x = torch.fft.fft(emb_x_[1]).real # FFT on pooler_output
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

#    def smooth_word_emb(self,
#        input_ids=None,
#        attention_mask=None,
#        token_type_ids=None,
#        position_ids=None,
#        head_mask=None,
#        inputs_embeds=None,
#        encoder_hidden_states=None,
#        encoder_attention_mask=None,
#        past_key_values=None,
#        use_cache=None,
#        output_attentions=None,
#        output_hidden_states=None,
#        return_dict=None, 
#        prune_mask=None,):
#
#        self.config = self.enc.encoder.config
#        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#        output_hidden_states = (
#            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#        )
#        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#        if self.config.is_decoder:
#            use_cache = use_cache if use_cache is not None else self.config.use_cache
#        else:
#            use_cache = False
#
#        if input_ids is not None and inputs_embeds is not None:
#            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#        elif input_ids is not None:
#            input_shape = input_ids.size()
#        elif inputs_embeds is not None:
#            input_shape = inputs_embeds.size()[:-1]
#        else:
#            raise ValueError("You have to specify either input_ids or inputs_embeds")
#
#        batch_size, seq_length = input_shape
#        device = input_ids.device if input_ids is not None else inputs_embeds.device
#
#        # past_key_values_length
#        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
#
#        if attention_mask is None:
#            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
#
#        if token_type_ids is None:
#            if hasattr(self.enc.encoder.embeddings, "token_type_ids"):
#                buffered_token_type_ids = self.enc.encoder.token_type_ids[:, :seq_length]
#                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
#                token_type_ids = buffered_token_type_ids_expanded
#            else:
#                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
#
#        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#        # ourselves in which case we just need to make it broadcastable to all heads.
#        extended_attention_mask: torch.Tensor = self.enc.encoder.get_extended_attention_mask(attention_mask, input_shape, device)
#
#        # If a 2D or 3D attention mask is provided for the cross-attention
#        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
#        if self.config.is_decoder and encoder_hidden_states is not None:
#            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#            if encoder_attention_mask is None:
#                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
#            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#        else:
#            encoder_extended_attention_mask = None
#
#        # Prepare head mask if needed
#        # 1.0 in head_mask indicate we keep the head
#        # attention_probs has shape bsz x n_heads x N x N
#        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#        head_mask = self.enc.encoder.get_head_mask(head_mask, self.config.num_hidden_layers)
#
#        embedding_output = self.enc.encoder.embeddings(
#            input_ids=input_ids,
#            position_ids=position_ids,
#            token_type_ids=token_type_ids,
#            inputs_embeds=inputs_embeds,
#            past_key_values_length=past_key_values_length,
#        )
#
#        if self.smooth_grad:
#            randn_noise = torch.randn_like(embedding_output)*self.noise_eps
#            p_emb_out = (embedding_output+randn_noise)*prune_mask.float()
#
#        else:
#            p_emb_out = embedding_output*prune_mask.float()
#
#        outputs = [p_emb_out, extended_attention_mask, head_mask, encoder_hidden_states, 
#                encoder_extended_attention_mask, past_key_values, use_cache, 
#                output_attentions, output_hidden_states, return_dict]
#
#        return outputs
#
#
#    def encoder_embedding(self,
#        p_emb_out=None,
#        extended_attention_mask=None,
#        head_mask=None,
#        encoder_hidden_states=None,
#        encoder_extended_attention_mask=None,
#        past_key_values=None,
#        use_cache=None,
#        output_attentions=None,
#        output_hidden_states=None,
#        return_dict=None, ):
#
#        encoder_outputs = self.enc.encoder.encoder(
#            p_emb_out,
#            attention_mask=extended_attention_mask,
#            head_mask=head_mask,
#            encoder_hidden_states=encoder_hidden_states,
#            encoder_attention_mask=encoder_extended_attention_mask,
#            past_key_values=past_key_values,
#            use_cache=use_cache,
#            output_attentions=output_attentions,
#            output_hidden_states=output_hidden_states,
#            return_dict=return_dict,
#        )
#        sequence_output = encoder_outputs[0]
#        pooled_output = self.enc.encoder.pooler(sequence_output) if self.enc.encoder.pooler is not None else None
#
#        output =  BaseModelOutputWithPoolingAndCrossAttentions(
#            last_hidden_state=sequence_output,
#            pooler_output=pooled_output,
#            past_key_values=encoder_outputs.past_key_values,
#            hidden_states=encoder_outputs.hidden_states,
#            attentions=encoder_outputs.attentions,
#            cross_attentions=encoder_outputs.cross_attentions,)
#
#        return output
#
