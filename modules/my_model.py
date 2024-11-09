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
    
    def predict_strategy(self, logits, encoded_info: dict):
        assert not self.training
        if Config.KNOWLEDGE_NAME == 'none':
            if Config.DATA_NAME == 'esconv':
                logits = logits[:, 0, -8:]
            elif Config.DATA_NAME == 'mi':
                logits = logits[:, 0, -10:]
        elif Config.KNOWLEDGE_NAME == 'basic':
            if Config.DATA_NAME == 'esconv':
                logits = logits[:, 0, -13:-5]
            elif Config.DATA_NAME == 'mi':
                logits = logits[:, 0, -15:-5]
        elif Config.KNOWLEDGE_NAME == 'bm25':
            if Config.DATA_NAME == 'esconv':
                logits = logits[:, 0, -9:-1]
            elif Config.DATA_NAME == 'mi':
                logits = logits[:, 0, -11:-1]
        elif Config.KNOWLEDGE_NAME == 'oracle':
            if Config.DATA_NAME == 'esconv':
                logits = logits[:, 0, -14:-6]
            elif Config.DATA_NAME == 'mi':
                logits = logits[:, 0, -16:-6]
        elif Config.KNOWLEDGE_NAME in ['sbert','graph']:
            if Config.DATA_NAME == 'esconv':
                logits = logits[:, 0, -16:-8]
            elif Config.DATA_NAME == 'mi':
                logits = logits[:, 0, -18:-8]
    
        strat_id = encoded_info.get('strat_id', None)
        if strat_id is not None:
            pred = strat_id
        else:
            pred = torch.argmax(logits, dim=-1)
        
        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=3, dim=-1)[1]
    
        encoded_info.update({
            'pred_strat_id': pred,
            'pred_strat_id_top1': pred_top1,
            'pred_strat_id_top3': pred_top3,
            'pred_strat_id_dist': F.softmax(logits, dim=-1)
        })

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        return_dict=None,
        **kwargs
    ):
        assert not self.training
        assert self.tokenizer is not None
        
        encoded_info = kwargs
        assert decoder_input_ids.size(1) == 1
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        self.predict_strategy(lm_logits, encoded_info)
        
        if Config.KNOWLEDGE_NAME == 'none':
            if Config.DATA_NAME == 'esconv':
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.tokenizer) - 8], 
                    dim=-1
                )
            elif Config.DATA_NAME == 'mi':
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.tokenizer) - 10], 
                    dim=-1
                )
        elif Config.KNOWLEDGE_NAME == 'basic':
            if Config.DATA_NAME == 'esconv':
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.tokenizer) - 13], 
                    dim=-1
                )
            elif Config.DATA_NAME == 'mi':
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.tokenizer) - 15], 
                    dim=-1
                )
        elif Config.KNOWLEDGE_NAME == 'bm25':
            if Config.DATA_NAME == 'esconv':
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.tokenizer) - 9], 
                    dim=-1
                )
            elif Config.DATA_NAME == 'mi':
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.tokenizer) - 11], 
                    dim=-1
                )
        elif Config.KNOWLEDGE_NAME == 'oracle':
            if Config.DATA_NAME == 'esconv':
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.tokenizer) - 14], 
                    dim=-1
                )
            elif Config.DATA_NAME == 'mi':
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.tokenizer) - 16], 
                    dim=-1
                )
        elif Config.KNOWLEDGE_NAME in ['sbert','graph']:
            if Config.DATA_NAME == 'esconv':
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.tokenizer) - 16], 
                    dim=-1
                )
            elif Config.DATA_NAME == 'mi':
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.tokenizer) - 18], 
                    dim=-1
                )
        
        assert 'max_length' in kwargs
        kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        kwargs['use_cache'] = True
        
        if len(self.tokenizer) > self.tokenizer.vocab_size:
            bad_words_ids = [[i] for i in range(self.tokenizer.vocab_size, len(self.tokenizer))]
            kwargs['bad_words_ids'] = bad_words_ids
        
        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        return encoded_info, generations[:, decoder_input_ids.size(1):]

    def tie_tokenizer(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if len(self.tokenizer) > self.tokenizer.vocab_size:
            self.resize_token_embeddings(len(self.tokenizer))
