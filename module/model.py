import torch
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import (
    Seq2SeqModelOutput,
    BaseModelOutput,
    Seq2SeqLMOutput,
)
from libs.config import Config, BlenderbotConfig as MyBlenderbotConfig
from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss


if "small" in MyBlenderbotConfig.PRETRAIN_MODEL.lower():
    from transformers.models.blenderbot_small import (
        BlenderbotSmallConfig as BlenderbotConfig,
        BlenderbotSmallForConditionalGeneration as BlenderbotModel,
    )
else:
    from transformers.models.blenderbot import (
        BlenderbotConfig,
        BlenderbotForConditionalGeneration as BlenderbotModel,
    )


class MyModel(BlenderbotModel):
    def __init__(self, config: BlenderbotConfig):
        super().__init__(config)
        self.tokenizer: PreTrainedTokenizer = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

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

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        return_dict=True,
        **kwargs
    ):
        kwargs.update(
            {
                "min_length": 100,
                "do_sample": True,
                "temperature": 0.7,
                "top_k": 30,
                "top_p": 0.3,
                "num_beams": 1,
                "num_return_sequences": 1,
                "length_penalty": 1.0,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 3,
                "encoder_no_repeat_ngram_size": 3,
                "pad_token_id": (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id
                    else self.tokenizer.eos_token_id
                ),
                "bos_token_id": (
                    self.tokenizer.bos_token_id
                    if self.tokenizer.bos_token_id
                    else self.tokenizer.cls_token_id
                ),
                "eos_token_id": (
                    self.tokenizer.eos_token_id
                    if self.tokenizer.eos_token_id
                    else self.tokenizer.sep_token_id
                ),
            }
        )
        encoded_info = kwargs

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
        lm_logits = (
            self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
        )

        if Config.DATA_NAME == "esconv":
            logits = lm_logits[:, 0, -8:]
        elif Config.DATA_NAME == "mi":
            logits = lm_logits[:, 0, -10:]

        pred = torch.argmax(logits, dim=-1)

        if Config.DATA_NAME == "esconv":
            decoder_input_ids = torch.cat(
                [decoder_input_ids, pred[..., None] + len(self.tokenizer) - 8], dim=-1
            )
        elif Config.DATA_NAME == "mi":
            decoder_input_ids = torch.cat(
                [decoder_input_ids, pred[..., None] + len(self.tokenizer) - 10], dim=-1
            )

        if len(self.tokenizer) > self.tokenizer.vocab_size:
            bad_words_ids = [
                [i] for i in range(self.tokenizer.vocab_size, len(self.tokenizer))
            ]
            kwargs["bad_words_ids"] = bad_words_ids

        generations = super().generate(
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        return encoded_info, generations[:, decoder_input_ids.size(1) :]

    def tie_tokenizer(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if len(self.tokenizer) > self.tokenizer.vocab_size:
            self.resize_token_embeddings(len(self.tokenizer))

    def aug(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
        **kwargs
    ):
        assert self.tokenizer is not None

        output_attentions = self.model.config.output_attentions
        output_hidden_states = self.model.config.output_hidden_states

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

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


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
