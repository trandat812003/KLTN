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
        
    def _read_file(self, file_name: str):
        with open(file_name, 'r', encoding='utf-8') as f:
            reader = f.readlines()

        return reader
    
    def setup(self):
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

    def _convert_inputs_to_features(self, inputs: list):
        if len(inputs) == 0:
            return []
        
        features = []
        for i in range(len(inputs)):
            ipt = inputs[i]
            feat = self._featurize(**ipt)
            features.append(feat)

        return features
    
    def _featurize(self, context, knowledge, persona, response, strat_id):
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
        bos = self.tokenizer.bos_token_id
        if bos is None:
            bos = self.tokenizer.cls_token_id
            assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.sep_token_id
            assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

        # breakpoint()

        context = [c + [eos] for c in context]
        context += [knowledge + [eos]]
        input_ids = sum(context, [])[:-1]
        input_ids = input_ids[-self.max_input_length:]
        persona_input_ids = persona
        
        labels = ([strat_id] + response + [eos])[:self.max_decoder_input_length + 1]
        decoder_input_ids = [bos] + labels[:-1]
        
        assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]

        return InputFeature(input_ids, decoder_input_ids, labels, persona_input_ids)

    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index: int) -> any:
        return self.data_list[index]
    
    def _convert_data_to_inputs(self, *args):
        pass
