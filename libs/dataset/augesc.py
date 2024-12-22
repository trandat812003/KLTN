import json
from transformers.tokenization_utils import PreTrainedTokenizer
from libs.config import Config
from libs.utils.input_feature import InputFeature
from libs.dataset.base import BaseDataset
from libs.config import Config


class AugDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, stage: str) -> None:
        super().__init__(tokenizer, stage)

    def _convert_data_to_inputs(self, data: list) -> list[dict]:
        process = lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))
        inputs, context, = [], []
        for d in data:
            text = process(self._norm(d[1]))
            if d[0] == 'sys':
                inputs.append({
                    'context': context.copy(),
                    'response': text,
                })

                self.inputs.append({
                    'context': [self.tokenizer.decode(c) for c in context.copy()],
                    'response': self.tokenizer.decode(text),
                })

            context = context + [text]

        return inputs
    
    def _featurize(self, context: list[int], response: list[int]) -> InputFeature:
        bos = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        eos = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id

        if not all([bos, eos]):
            raise ValueError("Token IDs (bos, eos) must be defined.")

        context = [c + [eos] for c in context]
        input_ids = sum(context, [])[-Config.MAX_INPUT_LENGTH:]

        labels = (response + [eos])[:Config.MAX_DECODER_INPUT_LENGTH + 1]
        decoder_input_ids = [bos] + labels[:-1]

        assert len(decoder_input_ids) == len(labels), "Mismatch between decoder inputs and labels"

        return InputFeature(input_ids, decoder_input_ids, labels)

    def _norm(self, s: str) -> str:
        return ' '.join(s.strip().split())
