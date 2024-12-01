import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import Seq2SeqModelOutput
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
        persona_input_ids=None,
        persona_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        **kwargs
    ):
        assert self.tokenizer is not None

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        output_attentions = self.model.config.output_attentions
        output_hidden_states = self.model.config.output_hidden_states
        
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
        context = self.persona_norm(context + persona_encoder_outputs.last_hidden_state)
        persona = self.context_norm(encoder_outputs.last_hidden_state + persona)

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

        return lm_logits

    def tie_tokenizer(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if len(self.tokenizer) > self.tokenizer.vocab_size:
            self.resize_token_embeddings(len(self.tokenizer))
