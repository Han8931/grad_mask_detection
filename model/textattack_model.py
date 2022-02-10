import torch
from torch.nn import CrossEntropyLoss
import transformers
import textattack

import pdb
from abc import ABC, abstractmethod
from model.train import batch_len

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

import spacy
from spacy.lang.en import English
import re

torch.cuda.empty_cache()


def print_function(args, f_name, batch_idx, num_successes, num_failed, num_skipped):
    print("="*100, flush=True)
    print(f"Model: {args.load_model} || FileName: {f_name} || Attack: {args.attack_method}", flush=True) 
    print(f"Multi_Mask: {args.multi_mask} || Sparsity: {args.p_prune} || SmoothGrad: {args.smooth_grad} ", flush=True) 
    print(f"Trials: {batch_idx+1} || # Success: {num_successes} || # Failed: {num_failed} || # Skipped: {num_skipped}", flush=True) 
    print(f"ASR: {num_successes/(batch_idx+1):.4f} || AAcc: {num_failed/(batch_idx+1):.4f}", flush=True) 
    print("="*100, flush=True)



class ModelWrapper(ABC):
    """A model wrapper queries a model with a list of text inputs.
    Classification-based models return a list of lists, where each sublist
    represents the model's scores for a given input.
    Text-to-text models return a list of strings, where each string is the
    output – like a translation or summarization – for a given input.
    """

    @abstractmethod
    def __call__(self, text_input_list, **kwargs):
        raise NotImplementedError()

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens."""
        raise NotImplementedError()

    def _tokenize(self, inputs):
        """Helper method for `tokenize`"""
        raise NotImplementedError()

    def tokenize(self, inputs, strip_prefix=False):
        """Helper method that tokenizes input strings
        Args:
            inputs (list[str]): list of input strings
            strip_prefix (bool): If `True`, we strip auxiliary characters added to tokens as prefixes (e.g. "##" for BERT, "Ġ" for RoBERTa)
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        tokens = self._tokenize(inputs)
        if strip_prefix:
            # `aux_chars` are known auxiliary characters that are added to tokens
            strip_chars = ["##", "Ġ", "__"]
            # TODO: Find a better way to identify prefixes. These depend on the model, so cannot be resolved in ModelWrapper.

            def strip(s, chars):
                for c in chars:
                    s = s.replace(c, "")
                return s

            tokens = [[strip(t, strip_chars) for t in x] for x in tokens]

        return tokens

class PyTorchModelWrapper(ModelWrapper):
    """Loads a PyTorch model (`nn.Module`) and tokenizer.
    Args:
        model (torch.nn.Module): PyTorch model
        tokenizer: tokenizer whose output can be packed as a tensor and passed to the model.
            No type requirement, but most have `tokenizer` method that accepts list of strings.
    """

    def __init__(self, model, tokenizer):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"PyTorch model must be torch.nn.Module, got type {type(model)}"
            )

        self.model = model
        self.tokenizer = tokenizer

    def to(self, device):
        self.model.to(device)

    def __call__(self, text_input_list, batch_size=32):
        model_device = next(self.model.parameters()).device
        ids = self.tokenizer(text_input_list)
        ids = torch.tensor(ids).to(model_device)

        with torch.no_grad():
            outputs = textattack.shared.utils.batch_model_predict(
                self.model, ids, batch_size=batch_size
            )

        return outputs

    def get_grad(self, text_input, loss_fn=CrossEntropyLoss()):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
            loss_fn (torch.nn.Module): loss function. Default is `torch.nn.CrossEntropyLoss`
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )
        if not isinstance(loss_fn, torch.nn.Module):
            raise ValueError("Loss function must be of type `torch.nn.Module`.")

        self.model.train()

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        #emb_hook = embedding_layer.register_full_backward_hook(grad_hook)
        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        ids = self.tokenizer([text_input])
        ids = torch.tensor(ids).to(model_device)

        predictions = self.model(ids)

        output = predictions.argmax(dim=1)
        loss = loss_fn(predictions, output)
        loss.backward()

        # grad w.r.t to word embeddings
        grad = torch.transpose(emb_grads[0], 0, 1)[0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": ids[0].tolist(), "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [self.tokenizer.convert_ids_to_tokens(self.tokenizer(x)) for x in inputs]


class HuggingFaceModelWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer):
        assert isinstance(
            model, transformers.PreTrainedModel
        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."
        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]

class CustomWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, args):
#        assert isinstance(
#            model, transformers.PreTrainedModel
#        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."

        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.num_ensemble = args.num_ensemble
        self.ensemble = True if args.num_ensemble>0 else False

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = self.args.device #next(self.model.parameters()).device
        inputs_dict.to(model_device)

        input_ids = inputs_dict['input_ids']
        attention_mask = inputs_dict['attention_mask']

#        if self.ensemble:
#            ens_list = []
#            for _ in range(self.num_ensemble):
#                with torch.no_grad():
#                    outputs = self.model(input_ids, attention_mask)
#                logits_ = outputs['logits']
#                ens_list.append(logits_)
#            logits = torch.stack(ens_list).mean(dim=0)
#        else:
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        logits = outputs['logits']

        return logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.enc.encoder.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
#        input_dict.to(model_device)
#        predictions = self.model(**input_dict).logits

        input_dict.to(model_device)
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']

        predictions = self.model(input_ids, attention_mask)
        predictions = predictions['logits']


        try:
            labels = predictions.argmax(dim=1)
            output = self.model(input_ids, attention_mask, labels)
            loss = output['loss']
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]

class GratWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, args):
#        assert isinstance(
#            model, transformers.PreTrainedModel
#        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."

        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.mask_idx = args.mask_idx
        self.pad_idx = args.pad_idx
        self.multi_mask = args.multi_mask

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = self.args.device #next(self.model.parameters()).device
        inputs_dict.to(model_device)

        input_ids = inputs_dict['input_ids']
        attention_mask = inputs_dict['attention_mask']
        b_length = batch_len(input_ids, self.pad_idx)

        grad_out = self.get_grad(text_input_list[0])
        delta_grad = grad_out['gradient']
        delta_grad = torch.Tensor(delta_grad).to(model_device)
        norm_grad = torch.norm(delta_grad, p=2, dim=-1)
        val, indices_ = torch.topk(norm_grad[:b_length[0]], 10)

        last_token = b_length[0].item()-1 # [SEP]
        indices = [x.item() for x in indices_ if x != self.pad_idx and x != last_token]

        #indices = self.model.grad_detection_batch(input_ids, attention_mask, topk=5, pred=None, mask_filter=True)
        self.model.zero_grad()           

#        masked_ids = input_ids.clone()
#        for ids_, m_idx in zip(masked_ids, indices):
#            for j in range(self.multi_mask):
#                ids_[m_idx[j]] = self.mask_idx

#        for j in range(self.multi_mask):
#            for ids_ in masked_ids:
#                ids_[indices_[j]] = self.mask_idx

        masked_ids = input_ids.clone()
        masked_ids[0][indices[0]] = self.mask_idx


        with torch.no_grad():
            outputs = self.model(masked_ids, attention_mask)

#        if self.vocab is not None:
#            #input_txt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
#            input_ids, attention_mask = self.masked_text2ids(text_input_list[0])
#        else:
#            input_ids = inputs_dict['input_ids']
#            attention_mask = inputs_dict['attention_mask']
#
#        b_length = (input_ids != self.args.pad_idx).data.sum(dim=-1)
#
#        with torch.no_grad():
#            outputs = self.model(input_ids, attention_mask)
#

        logits = outputs['logits']

        return logits


    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.enc.encoder.get_input_embeddings()
        #embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']

        predictions = self.model(input_ids, attention_mask)
        predictions = predictions['logits']
        #predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            output = self.model(input_ids, attention_mask, labels)
            loss = output['loss']
            #loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove() # Remove Hook
        self.model.eval()

        output = {"ids": input_ids, "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]

class PGratWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, args):
#        assert isinstance(
#            model, transformers.PreTrainedModel
#        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."

        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.mask_idx = args.mask_idx
        self.pad_idx = args.pad_idx
        self.multi_mask = args.multi_mask

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = self.args.device #next(self.model.parameters()).device
        inputs_dict.to(model_device)

        input_ids = inputs_dict['input_ids']
        attention_mask = inputs_dict['attention_mask']
        b_length = batch_len(input_ids, self.pad_idx)

#        with torch.no_grad():
#            outputs = self.model(input_ids, attention_mask)
#
#        pred = outputs['logits'].softmax(dim=-1).argmax(dim=-1) # prediction label
#        indices, delta_grad = self.model.grad_detection_batch2(input_ids, attention_mask, topk=5, pred=pred, mask_filter=True)
#        self.model.zero_grad()           
#
#        masked_ids = input_ids.clone()
#        for ids_, m_idx in zip(masked_ids, indices):
#            for j in range(self.multi_mask):
#                ids_[m_idx[j]] = self.mask_idx

        #output = model(masked_ids, attention_mask, labels, delta_grad)

        grad_out = self.get_grad(text_input_list[0])
        delta_grad = grad_out['gradient']
        delta_grad = torch.Tensor(delta_grad).to(model_device).detach()
        norm_grad = torch.norm(delta_grad, p=2, dim=-1)
        val, indices_ = torch.topk(norm_grad[:b_length[0]], 10)

        last_token = b_length[0].item()-1 # [SEP]
        indices = [x.item() for x in indices_ if x != self.pad_idx and x != last_token]

        self.model.zero_grad()           

        masked_ids = input_ids.clone()
        for j in range(self.multi_mask):
            for ids_ in masked_ids:
                ids_[indices_[j]] = self.mask_idx
                #ids_[indices_[j+1]] = self.mask_idx

        with torch.no_grad():
            outputs = self.model(masked_ids, attention_mask, delta_grad=delta_grad.unsqueeze(0))
            #outputs = self.model(input_ids, attention_mask)
            #outputs = self.model(input_ids, attention_mask, delta_grad=delta_grad.unsqueeze(0))

        logits = outputs['logits'].softmax(dim=-1)
#        print(logits)
#        print(f"Prediction: {logits.argmax(dim=-1)}") 

        return logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.enc.encoder.get_input_embeddings()
        #embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        #emb_hook = embedding_layer.register_backward_hook(grad_hook)
        emb_hook = embedding_layer.register_full_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']

        predictions = self.model(input_ids, attention_mask)
        predictions = predictions['logits']
        #predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            output = self.model(input_ids, attention_mask, labels)
            loss = output['loss']
            #loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove() # Remove Hook

        self.model.eval()

        output = {"ids": input_ids, "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]

class VATWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, args):
#        assert isinstance(
#            model, transformers.PreTrainedModel
#        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."

        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.mask_idx = args.mask_idx
        self.pad_idx = args.pad_idx
        self.multi_mask = args.multi_mask
        self.mask = args.mask

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = self.args.device #next(self.model.parameters()).device
        inputs_dict.to(model_device)

        input_ids = inputs_dict['input_ids']
        attention_mask = inputs_dict['attention_mask']
        b_length = batch_len(input_ids, self.pad_idx)

        grad_out = self.get_grad(text_input_list[0])
        delta_grad = grad_out['gradient']
        delta_grad = torch.Tensor(delta_grad).to(model_device).detach()
        self.model.zero_grad()           

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, delta_grad=delta_grad)

        logits = outputs['logits'].softmax(dim=-1)

        return logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.enc.encoder.get_input_embeddings()
        #embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        #emb_hook = embedding_layer.register_backward_hook(grad_hook)
        emb_hook = embedding_layer.register_full_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']

        predictions = self.model(input_ids, attention_mask)
        predictions = predictions['logits']
        #predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            output = self.model(input_ids, attention_mask, labels=labels)
            loss = output['loss']
            #loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove() # Remove Hook

        self.model.eval()

        output = {"ids": input_ids, "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]

class NoiseWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, args):
#        assert isinstance(
#            model, transformers.PreTrainedModel
#        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."

        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.mask_idx = args.mask_idx
        self.pad_idx = args.pad_idx
        self.multi_mask = args.multi_mask
        self.mask = args.mask

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = self.args.device #next(self.model.parameters()).device
        inputs_dict.to(model_device)

        input_ids = inputs_dict['input_ids']
        attention_mask = inputs_dict['attention_mask']
        b_length = batch_len(input_ids, self.pad_idx)

        if self.mask:
            grad_out = self.get_grad(text_input_list[0])
            delta_grad = grad_out['gradient']
            delta_grad = torch.Tensor(delta_grad).to(model_device).detach()
            norm_grad = torch.norm(delta_grad, p=2, dim=-1)
            val, indices_ = torch.topk(norm_grad[:b_length[0]], 10)

            last_token = b_length[0].item()-1 # [SEP]
            indices = [x.item() for x in indices_ if x != self.pad_idx and x != last_token]
            self.model.zero_grad()           

            masked_ids = input_ids.clone()
            for j in range(self.multi_mask):
                for ids_ in masked_ids:
                    ids_[indices_[j]] = self.mask_idx
                    #ids_[indices_[j+1]] = self.mask_idx

            with torch.no_grad():
                outputs = self.model(masked_ids, attention_mask)
        else:
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)

        logits = outputs['logits'].softmax(dim=-1)

        return logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.enc.encoder.get_input_embeddings()
        #embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        #emb_hook = embedding_layer.register_backward_hook(grad_hook)
        emb_hook = embedding_layer.register_full_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']

        predictions = self.model(input_ids, attention_mask)
        predictions = predictions['logits']
        #predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            output = self.model(input_ids, attention_mask, labels)
            loss = output['loss']
            #loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove() # Remove Hook

        self.model.eval()

        output = {"ids": input_ids, "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]

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

class SpacyWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, args):
#        assert isinstance(
#            model, transformers.PreTrainedModel
#        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."

        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.num_ensemble = args.num_ensemble
        self.ensemble = True if args.num_ensemble>0 else False

        nlp = English()
        self.spacy_tokenizer = nlp.tokenizer

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.

        txt_list = []
        for txt in text_input_list:
            clean = clean_str(txt, tokenizer=self.spacy_tokenizer)
            txt_list.append(" ".join(clean))

        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            txt_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
#        inputs_dict = self.tokenizer(
#            text_input_list,
#            add_special_tokens=True,
#            padding="max_length",
#            max_length=max_length,
#            truncation=True,
#            return_tensors="pt",
#        )

        model_device = self.args.device #next(self.model.parameters()).device
        inputs_dict.to(model_device)

        input_ids = inputs_dict['input_ids']
        attention_mask = inputs_dict['attention_mask']

#        if self.ensemble:
#            ens_list = []
#            for _ in range(self.num_ensemble):
#                with torch.no_grad():
#                    outputs = self.model(input_ids, attention_mask)
#                logits_ = outputs['logits']
#                ens_list.append(logits_)
#            logits = torch.stack(ens_list).mean(dim=0)
#        else:
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        logits = outputs['logits']

        return logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.enc.encoder.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
#        input_dict.to(model_device)
#        predictions = self.model(**input_dict).logits

        input_dict.to(model_device)
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']

        predictions = self.model(input_ids, attention_mask)
        predictions = predictions['logits']


        try:
            labels = predictions.argmax(dim=1)
            output = self.model(input_ids, attention_mask, labels)
            loss = output['loss']
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]

class SpacyMNLIWrapper(PyTorchModelWrapper):
    """Loads a HuggingFace ``transformers`` model and tokenizer."""

    def __init__(self, model, tokenizer, args):
#        assert isinstance(
#            model, transformers.PreTrainedModel
#        ), f"`model` must be of type `transformers.PreTrainedModel`, but got type {type(model)}."

        assert isinstance(
            tokenizer,
            (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast),
        ), f"`tokenizer` must of type `transformers.PreTrainedTokenizer` or `transformers.PreTrainedTokenizerFast`, but got type {type(tokenizer)}."

        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.num_ensemble = args.num_ensemble
        self.ensemble = True if args.num_ensemble>0 else False

        nlp = English()
        self.spacy_tokenizer = nlp.tokenizer

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.
        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        # Default max length is set to be int(1e30), so we force 512 to enable batching.

        txt_list = []
        for txt in text_input_list:
            txt = txt[0].rstrip()+" "+txt[1].lstrip()
            clean = clean_str(txt, tokenizer=self.spacy_tokenizer)
            txt_list.append(" ".join(clean))

        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            txt_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
#        inputs_dict = self.tokenizer(
#            text_input_list,
#            add_special_tokens=True,
#            padding="max_length",
#            max_length=max_length,
#            truncation=True,
#            return_tensors="pt",
#        )

        model_device = self.args.device #next(self.model.parameters()).device
        inputs_dict.to(model_device)

        input_ids = inputs_dict['input_ids']
        attention_mask = inputs_dict['attention_mask']

#        if self.ensemble:
#            ens_list = []
#            for _ in range(self.num_ensemble):
#                with torch.no_grad():
#                    outputs = self.model(input_ids, attention_mask)
#                logits_ = outputs['logits']
#                ens_list.append(logits_)
#            logits = torch.stack(ens_list).mean(dim=0)
#        else:
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        logits = outputs['logits']

        return logits

    def get_grad(self, text_input):
        """Get gradient of loss with respect to input tokens.
        Args:
            text_input (str): input string
        Returns:
            Dict of ids, tokens, and gradient as numpy array.
        """
        if isinstance(self.model, textattack.models.helpers.T5ForTextToText):
            raise NotImplementedError(
                "`get_grads` for T5FotTextToText has not been implemented yet."
            )

        self.model.train()
        embedding_layer = self.model.enc.encoder.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
#        input_dict.to(model_device)
#        predictions = self.model(**input_dict).logits

        input_dict.to(model_device)
        input_ids = input_dict['input_ids']
        attention_mask = input_dict['attention_mask']

        predictions = self.model(input_ids, attention_mask)
        predictions = predictions['logits']


        try:
            labels = predictions.argmax(dim=1)
            output = self.model(input_ids, attention_mask, labels)
            loss = output['loss']
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output

    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
