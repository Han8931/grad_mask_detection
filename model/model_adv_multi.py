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

from model.train import LinearScheduler, masking_fn, batch_len
#from model.robust_train import onezero_encoder, max_loss

import math

from nltk.tokenize import word_tokenize

from abc import ABC, abstractmethod

def _forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
):
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    next_decoder_cache = () if use_cache else None
    for i, layer_module in enumerate(self.layer):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        if getattr(self.config, "gradient_checkpointing", False) and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                    "`use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs_ = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            if i%self.config.nth_layers==0:
                randn_noise = torch.randn_like(layer_outputs_[0])*self.config.noise_eps
                #p_emb_out = (embedding_output+randn_noise)*prune_mask.float()
                temp  = layer_outputs_[0]+randn_noise
                layer_outputs = tuple(temp[None,:])
            else:
                layer_outputs = layer_outputs_

        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )

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


class GradModelWrapper(ABC):

    @abstractmethod
    def forward(self):
        pass

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

        emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)
        #emb_bn_x = self.batchnorm(emb_x)
        logits = self.cls(emb_x) # output from a FC layer

        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

        loss.backward()

        # grad w.r.t to word embeddings
        #grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove() # Remove Hook

        return emb_grads

class VAT2(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(VAT2, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device

        self.pad_idx = args.pad_idx
        self.mask_idx = args.mask_idx
        self.cls_token = args.cls_token
        self.sep_token = args.sep_token

        self.noise_eps = args.noise_eps
        self.noise_op = args.noise_op

    def get_last_grad(self, input_ids, attention_mask, labels):
        """ Construct FGSM adversarial examples on the examples X"""

        with torch.enable_grad():
            output = self.forward(input_ids, attention_mask, labels=labels)
            emb_last = output['emb']
            delta = torch.zeros_like(emb_last, requires_grad=True)

            logits = self.cls(emb_last+delta) # output from a FC layer
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))
        loss.backward()

        #return delta.grad.detach()
        return delta.grad.detach().sign() # FGSM style

    def forward(self, input_ids, attention_mask, delta_grad=None, labels=None):
        """
        - x: input
        """
        #b_length = batch_len(input_ids, self.pad_idx)
        #emb_x_ = self.emb_prune(input_ids, attention_mask)

        emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)

        if delta_grad is not None:
            if self.noise_op == "add":
                logits = self.cls(emb_x+delta_grad*self.noise_eps) # output from a FC layer
            elif self.noise_op == "mult":
                noise = delta_grad*self.noise_eps+1
                logits = self.cls(emb_x*noise) # output from a FC layer
            pdb.set_trace() 
        else:
            logits = self.cls(emb_x) # output from a FC layer

        
        if labels is not None:

            loss_fn = nn.CrossEntropyLoss()
            loss_ce = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

            output = {'logits': logits, 'loss': loss_ce, 'emb': emb_x}

            return output

        else:
            output = {'logits': logits, 'emb': emb_x}

        return output

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

        emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)
        #emb_bn_x = self.batchnorm(emb_x)
        logits = self.cls(emb_x) # output from a FC layer

        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

        loss.backward()

        # grad w.r.t to word embeddings
        #grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove() # Remove Hook

        return emb_grads

class VAT(GradModelWrapper, nn.Module):
    def __init__(self, encoder, classifier, args):
        super(VAT, self).__init__()
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
        self.noise_eps = args.noise_eps



    def forward(self, input_ids, attention_mask, delta_grad=None, labels=None):
        """
        - x: input
        """
        #b_length = batch_len(input_ids, self.pad_idx)
        #emb_x_ = self.emb_prune(input_ids, attention_mask)

        if delta_grad is not None:

            embedding_layer = self.enc.encoder.get_input_embeddings()
            #emb_x_ = self.enc.encoder(input_ids, attention_mask)

            def grad_hook(module, inputs, outputs):
                return outputs+delta_grad*self.noise_eps

            emb_hook = embedding_layer.register_forward_hook(grad_hook)

        emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)

        #emb_bn_x = self.batchnorm(emb_x)
        logits = self.cls(emb_x) # output from a FC layer

        if delta_grad is not None:
            emb_hook.remove() # Remove Hook

        
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
        return_dict=None, ):

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

        embedding_output = self.enc.encoder.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        norm_emb = torch.abs(embedding_output)
        val, indices_ = torch.topk(norm_emb, self.p_topk)
        pruned_emb = torch.scatter(norm_emb, 2, indices_, 0)
        prune_mask = pruned_emb!=0

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

class NoiseMaskAdvTr(GradModelWrapper, nn.Module):
    def __init__(self, encoder, classifier, args):
        super(NoiseMaskAdvTr, self).__init__()
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
        self.nth_layers = args.nth_layers

    def forward(self, input_ids, attention_mask, labels=None):
        """
        - x: input
        """
        #b_length = batch_len(input_ids, self.pad_idx)
        #emb_x_ = self.emb_prune(input_ids, attention_mask)

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
        return_dict=None, ):

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

        embedding_output = self.enc.encoder.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        norm_emb = torch.abs(embedding_output)
        val, indices_ = torch.topk(norm_emb, self.p_topk)
        pruned_emb = torch.scatter(norm_emb, 2, indices_, 0)
        prune_mask = pruned_emb!=0

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


class NoiseAdvTr(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(NoiseAdvTr, self).__init__()
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
        #b_length = batch_len(input_ids, self.pad_idx)

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


class PGradAdvTr(GradModelWrapper, nn.Module):
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

#    def embeddings(self, input_ids, attention_mask):
#        outputs = self.enc.encoder(input_ids, attention_mask, output_hidden_states=True)
#        embs = outputs['hidden_states']
#        emb_init = embs[0]
#        emb_last = embs[-1]
#
#        return emb_last, emb_init

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

class NoiseSingle2(nn.Module):
    """
    Noise at last embedding [CLS]
    """
    def __init__(self, encoder, classifier, args):
        super(NoiseSingle2, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device
        self.pad_idx = args.pad_idx
        self.mask_idx = args.mask_idx

        self.noise_eps = args.noise_eps
        self.noise_op = args.noise_op
        #self.batchnorm = nn.BatchNorm1d(768, affine=False)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        - x: input
        """
        emb_x_ = self.enc.encoder(input_ids, attention_mask)

        hidden_state = emb_x_[0]  # (bs, seq_len, dim)
        emb_x = hidden_state[:, 0]  # (bs, dim)

        #emb_bn_x = self.batchnorm(emb_x)
        if self.noise_op == "add":
            randn_noise = torch.randn_like(emb_x)*self.noise_eps
            logits = self.cls(emb_x+randn_noise) # output from a FC layer
        elif self.noise_op == "mult":
            randn_noise = torch.randn_like(emb_x)*self.noise_eps+1
            logits = self.cls(emb_x*randn_noise) # output from a FC layer

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

class NoiseSingle(nn.Module):
    def __init__(self, encoder, classifier, args):
        super(NoiseSingle, self).__init__()
        self.enc = encoder
        self.cls = classifier
        self.num_classes = args.num_classes
        self.device = args.device
        self.pad_idx = args.pad_idx
        self.mask_idx = args.mask_idx

        self.noise_eps = args.noise_eps
        self.noise_op = args.noise_op
        #self.batchnorm = nn.BatchNorm1d(768, affine=False)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        - x: input
        """
        embedding_layer = self.enc.encoder.get_input_embeddings()

        def grad_hook(module, inputs, outputs):
            randn_noise = torch.randn_like(outputs)*self.noise_eps
            return outputs+randn_noise

        #emb_hook = embedding_layer.register_backward_hook(grad_hook)
        emb_hook = embedding_layer.register_forward_hook(grad_hook)

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

    def embeddings(self, input_ids, attention_mask):
        outputs = self.enc.encoder(input_ids, attention_mask, output_hidden_states=True)
        embs = outputs['hidden_states']
        emb_init = embs[0]
        emb_last = embs[-1]

        return emb_last, emb_init

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

    def encoding(self, x):
        return self.enc(x)

    def encoding_tokens_norm(self, input_ids):
        src_mask_1 = (input_ids!=0)
        #src_mask_2 = (input_ids!=102)
        #src_mask_3 = (input_ids!=101)

        src_mask = src_mask_1 #& src_mask_2 & src_mask_3
        src_mask = src_mask==False

        enc_out = self.enc.encoder(input_ids)
        hidden_state = enc_out[0]  # (bs, seq_len, dim)
        norm = torch.norm(hidden_state, p=2, dim=-1)
        norm = norm.masked_fill(src_mask, 0)
        return norm, src_mask

    def encoding_tokens_norm_min(self, input_ids):
        src_mask_1 = (input_ids!=0)
        src_mask_2 = (input_ids!=102)
        src_mask_3 = (input_ids!=101)

        src_mask = src_mask_1 & src_mask_2 & src_mask_3
        src_mask = src_mask==False

        enc_out = self.enc.encoder(input_ids)
        hidden_state = enc_out[0]  # (bs, seq_len, dim)
        norm = torch.norm(hidden_state, p=2, dim=-1)
        norm = norm.masked_fill(src_mask, 10000)
        return norm, src_mask


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


