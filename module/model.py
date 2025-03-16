import torch
from transformers import PreTrainedTokenizer
from transformers.modeling_outputs import Seq2SeqModelOutput
from libs.config import Config, BlenderbotConfig as MyBlenderbotConfig

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
                "max_length": Config.MAX_INPUT_LENGTH,
                "min_length": 15,
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

        kwargs["max_length"] = Config.MAX_INPUT_LENGTH + decoder_input_ids.size(1)
        kwargs["use_cache"] = True

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
