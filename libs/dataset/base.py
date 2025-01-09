import json
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import Dataset
from libs.config import Config
from libs.utils.input_feature import InputFeature
from libs.utils.file_manager import save_file_pickle, load_file_pickle, read_file


class BaseDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, stage: str) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_input_length = 512
        self.max_decoder_input_length = 50
        self.stage = stage
        self.data_list = []
        self.inputs = []

    def setup(self) -> None:
        self.data_list, self.inputs = load_file_pickle(
            root_file=f'./.cache/dataset/{Config.DATA_NAME}.{Config.BASELINE}',
            file_name=f'{self.stage}.pkl'
        )

        if self.data_list is not None:
            return
        
        self.data_list = []
        self.inputs = []
        reader = read_file(f'./dataset/{Config.DATA_NAME}/{Config.BASELINE}/{self.stage}.txt')

        for line in reader:
            data = json.loads(line)
            inputs = self._convert_data_to_inputs(data)
            features = self._convert_inputs_to_features(inputs)
            self.data_list.extend(features)

        save_file_pickle(
            f'./.cache/dataset/{Config.DATA_NAME}.{Config.BASELINE}/{self.stage}.pkl',
            {'data_list': self.data_list, 'inputs': self.inputs}
        )

    def _convert_inputs_to_features(self, inputs: list[dict]) -> list[InputFeature]:
        if not inputs:
            return []
        
        features = [self._featurize(**ipt) for ipt in inputs]
        return features
    
    def _featurize(self, context: list, persona, knowledge: list, response) -> InputFeature:
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        bos = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        eos = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id

        if not all([pad, bos, eos]):
            raise ValueError("Token IDs (pad, bos, eos) must be defined.")
        
        context = [c + [eos] for c in context]
        context += [knowledge + [eos]]
        input_ids = sum(context, [])[-self.max_input_length:]

        labels = (response + [eos])[:self.max_decoder_input_length + 1]
        decoder_input_ids = [bos] + labels[:-1]
        persona_input_ids = persona[-self.max_input_length:]

        return InputFeature(input_ids, decoder_input_ids, labels, persona_input_ids)
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index: int) -> InputFeature:
        return self.data_list[index]
    
    def _convert_data_to_inputs(self, *args) -> list[dict]:
        raise NotImplementedError("This method should be implemented in subclasses.")
