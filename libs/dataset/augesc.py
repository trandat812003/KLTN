import json
from transformers.tokenization_utils import PreTrainedTokenizer
from libs.config import Config
from libs.utils.input_feature import InputFeature
from libs.utils.file_manager import save_file_pickle, load_file_pickle, read_file
from libs.dataset.base import BaseDataset
from libs.config import Config


class AugDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, stage: str) -> None:
        super().__init__(tokenizer, stage)
        self.max_input_length = Config.MAX_INPUT_LENGTH
        self.max_decoder_input_length = Config.MAX_DECODER_INPUT_LENGTH
        self.data_list = []

    def setup(self) -> None:
        self.data_list = load_file_pickle(
            root_file=f'./.cache/dataset/aug_dataset',
            file_name=f'augesc.pkl'
        )

        if self.data_list is not None:
            return
        
        self.data_list = []
        reader = read_file(f'./data_aug/augesc.txt')

        for line in reader:
            data = json.loads(line)
            inputs = self._convert_data_to_inputs(data)
            features = self._convert_inputs_to_features(inputs)
            self.data_list.extend(features)

        save_file_pickle(
            f'./.cache/dataset/aug_dataset/augesc.pkl',
            self.data_list
        )

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

            context = context + [text]

        return inputs

    def _convert_inputs_to_features(self, inputs: list[dict]) -> list[InputFeature]:
        if not inputs:
            return []
        
        features = [self._featurize(**ipt) for ipt in inputs]
        return features
    
    def _featurize(self, context: list[int], response: list[int]) -> InputFeature:
        bos = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        eos = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id

        if not all([bos, eos]):
            raise ValueError("Token IDs (bos, eos) must be defined.")

        context = [c + [eos] for c in context]
        input_ids = sum(context, [])[-self.max_input_length:]

        labels = (response + [eos])[:self.max_decoder_input_length + 1]
        decoder_input_ids = [bos] + labels[:-1]

        assert len(decoder_input_ids) == len(labels), "Mismatch between decoder inputs and labels"

        return InputFeature(input_ids, decoder_input_ids, labels)
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index: int) -> InputFeature:
        return self.data_list[index]

    def _norm(self, s: str) -> str:
        return ' '.join(s.strip().split())
