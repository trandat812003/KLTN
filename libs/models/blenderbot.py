from transformers.models.blenderbot_small import (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration, )
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import PreTrainedTokenizer
import torch
import torch.nn.functional as F


class ModelBlenderbot(BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self._tokenizer = None
        self._knowledge_name: str
        self._data_name: str

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        validation=False,
        **kwargs
    ):
        assert self._tokenizer is not None
        assert (self.training or validation) == (labels is not None)
        if validation:
            labels[:, 0] = -100
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if not self.training and not validation:
            use_cache = True
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
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        
        if validation:
            lm_logits = lm_logits[..., :self._tokenizer.vocab_size].contiguous()

        masked_lm_loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
            loss = loss.view(labels.size(0), labels.size(1))
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
            ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

        if not self.training and not validation:
            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        elif self.training:
            assert not validation
            res = {'all': masked_lm_loss, 'ppl': ppl_value, }
            return res

        else:
            assert not self.training
            return loss, label_size

    def predict_strategy(self, logits, encoded_info):
        assert not self.training
        strat_id = encoded_info.get('strat_id', None)
        if self._knowledge_name == 'none':
            if self._data_name == 'esconv':
                logits = logits[:, 0, -8:]
            elif self._data_name == 'mi':
                logits = logits[:, 0, -10:]
        elif self._knowledge_name == 'basic':
            if self._data_name == 'esconv':
                logits = logits[:, 0, -13:-5]
            elif self._data_name == 'mi':
                logits = logits[:, 0, -15:-5]
        elif self._knowledge_name == 'bm25':
            if self._data_name == 'esconv':
                logits = logits[:, 0, -9:-1]
            elif self._data_name == 'mi':
                logits = logits[:, 0, -11:-1]
        elif self._knowledge_name == 'oracle':
            if self._data_name == 'esconv':
                logits = logits[:, 0, -14:-6]
            elif self._data_name == 'mi':
                logits = logits[:, 0, -16:-6]
        elif self._knowledge_name in ['sbert','graph']:
            if self._data_name == 'esconv':
                logits = logits[:, 0, -16:-8]
            elif self._data_name == 'mi':
                logits = logits[:, 0, -18:-8]

    
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
        assert self._tokenizer is not None
        
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
        
        if self._knowledge_name == 'none':
            if self._data_name == 'esconv':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self._tokenizer) - 8], dim=-1)
            elif self._data_name == 'mi':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self._tokenizer) - 10], dim=-1)
        elif self._knowledge_name == 'basic':
            if self._data_name == 'esconv':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self._tokenizer) - 13], dim=-1)
            elif self._data_name == 'mi':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self._tokenizer) - 15], dim=-1)
        elif self._knowledge_name == 'bm25':
            if self._data_name == 'esconv':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self._tokenizer) - 9], dim=-1)
            elif self._data_name == 'mi':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self._tokenizer) - 11], dim=-1)
        elif self._knowledge_name == 'oracle':
            if self._data_name == 'esconv':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self._tokenizer) - 14], dim=-1)
            elif self._data_name == 'mi':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self._tokenizer) - 16], dim=-1)
        elif self._knowledge_name in ['sbert','graph']:
            if self._data_name == 'esconv':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self._tokenizer) - 16], dim=-1)
            elif self._data_name == 'mi':
                decoder_input_ids = torch.cat([decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self._tokenizer) - 18], dim=-1)
        
        assert 'max_length' in kwargs
        kwargs['max_length'] = kwargs['max_length'] + decoder_input_ids.size(1)
        kwargs['use_cache'] = True
        
        if len(self._tokenizer) > self._tokenizer.vocab_size:
            bad_words_ids = [[i] for i in range(self._tokenizer.vocab_size, len(self._tokenizer))]
            kwargs['bad_words_ids'] = bad_words_ids
        
        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        return encoded_info, generations[:, decoder_input_ids.size(1):]

    def tie_tokenizer(self, tokenizer: PreTrainedTokenizer):
        self._tokenizer = tokenizer
        if len(self._tokenizer) > self._tokenizer.vocab_size:
            self.resize_token_embeddings(len(self._tokenizer))
    