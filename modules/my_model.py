import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.models.blenderbot_small import (BlenderbotSmallConfig, BlenderbotSmallForConditionalGeneration)
from typing import Any, Dict
from libs.config import Config


class MyModel(BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self.tokenizer: PreTrainedTokenizer = None

        self.personanorm = nn.LayerNorm(self.model.config.d_model, elementwise_affine=True)  # 512
        self.contextnorm = nn.LayerNorm(self.model.config.d_model, elementwise_affine=True)  # 512
        self.persona_context_w = nn.Parameter(torch.tensor([1 / 3, 1 / 3, 1 / 3]))
        self.strategy_alpha = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))

        self.generation_strategy = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        persona_input_ids=None,
        persona_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        **kwargs
    ):
        assert self.tokenizer is not None

        output_attentions = self.model.config.output_attentions
        output_hidden_states = self.model.config.output_hidden_states

        my_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        my_logits = self.lm_head(my_outputs[0]) + self.final_logits_bias
        
        if kwargs.get('predict', None) is None:
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            persona_encoder_outputs = self.model.encoder(
                input_ids=persona_input_ids,
                attention_mask=persona_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            context = torch.stack(
                [torch.matmul(torch.softmax(torch.matmul(j, i.t()), dim=-1), i) 
                    for i, j in zip(encoder_outputs[0], persona_encoder_outputs[0])]
            )
            persona = torch.stack(
                [torch.matmul(torch.softmax(torch.matmul(i, j.t()), dim=-1), j) 
                    for i, j in zip(encoder_outputs[0], persona_encoder_outputs[0])]
            )
            context = self.personanorm(context + persona_encoder_outputs.last_hidden_state)
            persona = self.contextnorm(encoder_outputs.last_hidden_state + persona)

            w1 = torch.exp(self.persona_context_w[0]) / torch.sum(torch.exp(self.persona_context_w))
            w2 = torch.exp(self.persona_context_w[1]) / torch.sum(torch.exp(self.persona_context_w))
            w3 = torch.exp(self.persona_context_w[2]) / torch.sum(torch.exp(self.persona_context_w))

            encoder_outputs.last_hidden_state = w1 * encoder_outputs.last_hidden_state + w2 * context + w3 * persona

            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            outputs = Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
            strat_ids = kwargs.get('strat_id', None)

            self.cal_strategy(strat_ids, lm_logits=lm_logits, my_logits=my_logits)
        
            return lm_logits
        else:
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

            self.cal_strategy(self.generation_strategy, lm_logits=lm_logits, my_logits=my_logits)

            return Seq2SeqLMOutput(
                # loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
        
    def cal_strategy(self, strat_ids, lm_logits, my_logits):
        alpha_l = []
        lm_size = lm_logits.size()
        for i in strat_ids:
            tmp_alpha = self.strategy_alpha[i.item()]
            tmp_alpha = tmp_alpha * torch.ones(lm_size[1], lm_size[2], device=self.device)
            alpha_l.append(tmp_alpha)
        alpha_l = torch.stack(alpha_l)
        lm_logits = (torch.ones_like(lm_logits, device=self.device) + alpha_l) * lm_logits - alpha_l * my_logits

        return lm_logits

    def predict_strategy(self, logits, encoded_info):
        assert not self.training
        strat_id = encoded_info.get('strat_id', None)
        logits = logits[:, 0, -8:]

        if strat_id is not None:
            pred = strat_id
        else:
            pred = torch.argmax(logits, dim=-1)

        pred_top1 = torch.topk(logits, k=1, dim=-1)[1]
        pred_top3 = torch.topk(logits, k=5, dim=-1)[1]

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
            persona_input_ids=None,
            persona_attention_mask=None,
            decoder_input_ids=None,
            return_dict=True,
            **kwargs
    ):
        kwargs.update({
            'predict': True,
            'other_res': {'acc_map': {'cls_strat_id': 'pred_strat_id'}, 'cls_strat_id': kwargs['strat_id']},
            'max_length': Config.MAX_DECODER_INPUT_LENGTH,
            'min_length': 10,
            'do_sample': True,
            'temperature': 0.5,
            'top_k': 30,
            'top_p': 0.9,
            'num_beams': 1,
            'num_return_sequences': 1,
            'length_penalty': 1.0,
            'repetition_penalty': 1.03,
            'no_repeat_ngram_size': 0,
            'encoder_no_repeat_ngram_size': 3,
            'pad_token_id': self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
            'bos_token_id': self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else  self.tokenizer.cls_token_id,
            'eos_token_id': self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else  self.tokenizer.sep_token_id,
        })
        encoded_info = kwargs
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        self.my_encoder_outputs = encoder_outputs.copy()

        persona_encoder_outputs = self.model.encoder(
            input_ids=persona_input_ids,
            attention_mask=persona_attention_mask,
            return_dict=return_dict,
        )
        context = torch.stack(
            [torch.matmul(torch.softmax(torch.matmul(j, i.t()), dim=-1), i) 
             for i, j in zip(encoder_outputs[0], persona_encoder_outputs[0])]
        )
        persona = torch.stack(
            [torch.matmul(torch.softmax(torch.matmul(i, j.t()), dim=-1), j) 
             for i, j in zip(encoder_outputs[0], persona_encoder_outputs[0])]
        )
        context = self.personanorm(context + persona_encoder_outputs.last_hidden_state)
        persona = self.contextnorm(encoder_outputs.last_hidden_state + persona)
        w1 = torch.exp(self.persona_context_w[0]) / torch.sum(torch.exp(self.persona_context_w))
        w2 = torch.exp(self.persona_context_w[1]) / torch.sum(torch.exp(self.persona_context_w))
        w3 = torch.exp(self.persona_context_w[2]) / torch.sum(torch.exp(self.persona_context_w))
        encoder_outputs.last_hidden_state = w1 * encoder_outputs.last_hidden_state + w2 * context + w3 * persona

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        my_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=self.my_encoder_outputs,
            return_dict=return_dict,
        )
        my_logits = self.lm_head(my_outputs[0]) + self.final_logits_bias
        lm_logits = (1+0.075)*lm_logits - 0.075*my_logits
        self.predict_strategy(lm_logits, encoded_info)
        self.generation_strategy = encoded_info['pred_strat_id']
        decoder_input_ids = torch.cat(
            [decoder_input_ids, encoded_info['pred_strat_id'][..., None] + len(self.tokenizer) - 8], 
            dim=-1
        )
        kwargs['max_length'] = Config.MAX_DECODER_INPUT_LENGTH + decoder_input_ids.size(1)
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
        encoded_info['persona'] = persona_input_ids
        return encoded_info, generations[:, decoder_input_ids.size(1):]
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs) -> Dict[str, Any]:
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to prepare inputs in the
        generate method.
        """
        if kwargs['predict']: 
            models_kwargs = super(MyModel, self).prepare_inputs_for_generation(input_ids, **kwargs)
            models_kwargs.update({'predict': True})
            return models_kwargs
        else:
            return super(MyModel, self).prepare_inputs_for_generation(input_ids, **kwargs)

    def tie_tokenizer(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if len(self.tokenizer) > self.tokenizer.vocab_size:
            self.resize_token_embeddings(len(self.tokenizer))
