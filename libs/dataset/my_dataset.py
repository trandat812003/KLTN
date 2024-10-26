import os
import pickle
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from transformers.tokenization_utils import PreTrainedTokenizer
from libs.utils import InputFeatures


class MyDataset(Dataset):
    def __init__(self, knowledge_name: str) -> None:
        self._knowledge_name = knowledge_name
        self._max_input_length = 256
        self._max_decoder_input_length = 40
        self._data: List = []
        self._tokenizer: PreTrainedTokenizer = None

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict:
        return self.collate(self._data[idx])

    def setup(self):
        raise NotImplemented()

    def convert_data_to_inputs(self, data: str):
        raise NotImplemented()

    def convert_inputs_to_feature(self, inputs: List):
        raise NotImplemented()
    
    def collate(feature: InputFeatures):
        raise NotImplemented()
    
    def _load_dataset_from_pkl(self, file_name):
        raise NotImplemented()

    def _save_dataset_to_pkl(self, dataset, pkl_path):
        raise NotImplemented()

    def setup_and_load_dataset(self, dataset, path, mode):
        if not os.path.exists(path):
            dataset.setup(mode)
            self._save_dataset_to_pkl(dataset, path)
        else:
            dataset = self._load_dataset_from_pkl(path)
        return dataset
    
    def convert_inputs_to_feature(self, inputs: list) -> list:
        if len(inputs) == 0:
            return []
        
        features = []
        for i in range(len(inputs)):
            ipt = inputs[i]
            feat = self._featurize(ipt['context'], ipt['knowledge'], ipt['response'], ipt['strat_id'])
            features.append(feat)
        return features
    
    def _featurize(self, context, knowledge, response, strat_id):
        bos = self._tokenizer.bos_token_id if self._tokenizer.bos_token_id is not None else self._tokenizer.cls_token_id
        eos = self._tokenizer.eos_token_id if self._tokenizer.eos_token_id is not None else self._tokenizer.sep_token_id

        context = [c + [eos] for c in context]
        context += [knowledge + [eos]]
        input_ids = sum(context, [])[:-1]
        input_ids = input_ids[-self._max_input_length:]
        
        labels = ([strat_id] + response + [eos])[:self._max_decoder_input_length + 1]
        decoder_input_ids = [bos] + labels[:-1]
        
        assert len(decoder_input_ids) == len(labels), decoder_input_ids[1:] == labels[:-1]

        return InputFeatures(
            input_ids,
            decoder_input_ids, 
            labels,
        )
    
    def collate(self, features: InputFeatures):
        pad = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f.input_ids, dtype=torch.long) for f in features],
            batch_first=True, 
            padding_value=pad
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
            batch_first=True, 
            padding_value=0.
        )

        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
            batch_first=True, 
            padding_value=pad
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f.labels, dtype=torch.long) for f in features],
            batch_first=True, 
            padding_value=-100
        )

        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            #'input_length': input_length,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
        }
        
        return res

