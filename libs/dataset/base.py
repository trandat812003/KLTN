import json
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import Dataset
from libs.config import Config
from libs.utils.input_feature import InputFeature
from libs.utils.save_and_load_pickle import save_file_pickle, load_file_pickle


class BaseDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, stage: str) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_input_length = 512
        self.max_decoder_input_length = 50
        self.stage = stage
        self.data_list = []

    def _read_file(self, file_name: str) -> list[str]:
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                return f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_name}")

    def setup(self) -> None:
        self.data_list = load_file_pickle(
            root_file=f'./.cache/dataset/{Config.DATA_NAME}.{Config.BASELINE}',
            file_name=f'{self.stage}.pkl'
        )

        if self.data_list is not None:
            return
        
        self.data_list = []
        reader = self._read_file(f'./dataset/{Config.DATA_NAME}/{Config.BASELINE}/{self.stage}.txt')

        for line in reader:
            data = json.loads(line)
            inputs = self._convert_data_to_inputs(data)
            features = self._convert_inputs_to_features(inputs)
            self.data_list.extend(features)

        save_file_pickle(
            f'./.cache/dataset/{Config.DATA_NAME}.{Config.BASELINE}/{self.stage}.pkl',
            self.data_list
        )

    def _convert_inputs_to_features(self, inputs: list[dict]) -> list[InputFeature]:
        if not inputs:
            return []
        
        features = [self._featurize(**ipt) for ipt in inputs]
        return features
    
    def _featurize(self, context: list, persona, knowledge: list, strat_id, last_text) -> InputFeature:
        process = lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        bos = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        eos = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id

        if not all([pad, bos, eos]):
            raise ValueError("Token IDs (pad, bos, eos) must be defined.")
        
        strat_id = process(strat_id)
        context = self.tokenizer(context).input_ids
        context += process(knowledge) + [eos]
        labels = (strat_id + context + [eos])[:self.max_decoder_input_length + 1]
        persona_input_ids = self.tokenizer(persona).input_ids
        input_ids = self.tokenizer('</s> <s>'.join(last_text)).input_ids
        input_ids = input_ids[-self.max_input_length:]
        decoder_input_ids = [bos] + labels[:-1]

        return InputFeature(input_ids, decoder_input_ids, labels, persona_input_ids)
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index: int) -> InputFeature:
        return self.data_list[index]
    
    def _convert_data_to_inputs(self, *args) -> list[dict]:
        raise NotImplementedError("This method should be implemented in subclasses.")
