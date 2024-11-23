# coding=utf-8
# copied from bart


import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers.models.blenderbot_small import (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration)
from libs.config import Config


class MyModel(BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self.tokenizer: PreTrainedTokenizer = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=None,
        return_dict=None,
        **kwargs
    ):
        assert self.tokenizer is not None
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        outputs = self.lm_head(outputs[0]) + self.final_logits_bias
        return outputs

    def tie_tokenizer(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if len(self.tokenizer) > self.tokenizer.vocab_size:
            self.resize_token_embeddings(len(self.tokenizer))
