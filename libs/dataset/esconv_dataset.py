import json
import os
import pickle
import torch
from typing import Dict
from libs.dataset.my_dataset import MyDataset
from libs.utils.get_tokenizer import get_tokenizer
from libs.utils import InputFeatures


class ESConvDataset(MyDataset):
    def __init__(self, knowledge_name: str) -> None:
        super().__init__(knowledge_name)

    def setup(self, stage: str):
        self._tokenizer = get_tokenizer("esconv", self._knowledge_name)

        data_path = f"./dataset/esconv/{self._knowledge_name}/{stage}.txt"
        with open(data_path,'r',encoding='utf-8') as f:
            reader = f.readlines()

        for line in reader:
            data = json.loads(line)

            inputs = self.convert_data_to_inputs(data)
            self._data.append(self.convert_inputs_to_feature(inputs))

    def convert_data_to_inputs(self, data: dict) -> list:
        process = lambda x: self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(x))

        dialog = data['dialog']
        inputs = []
        context = []
        knowledge = []
        
        for i in range(len(dialog)):
            text = self._norm(dialog[i]['text'])
            text = process(text)
            
            if dialog[i]['speaker'] == 'sys':
                strat_id = process('[' + dialog[i]['strategy'] + ']')
                assert len(strat_id) == 1
                strat_id = strat_id[0]
                
                if self._knowledge_name == 'oracle':
                    heal = process('[knowledge]') + process(dialog[i]['heal'])
                elif self._knowledge_name in ['sbert','graph']:
                    heal = process(dialog[i]['heal'])
                else:
                    heal = []
            else:
                if self._knowledge_name in ['basic','oracle','sbert','graph']:
                    knowledge = process(dialog[i]['knowledge'])
                elif self._knowledge_name == 'bm25':
                    knowledge = process('[knowledge]') + process(dialog[i]['knowledge'])
                else:
                    knowledge = []
            
            if i > 0 and dialog[i]['speaker'] == 'sys':
                res = {
                    'dialog': dialog,
                    'context': context.copy(),
                    'knowledge': knowledge + heal,
                    'response': text,
                    'strat_id': strat_id,
                }
                
                inputs.append(res)

            context = context + [text]

        return inputs
    
    def collate(self, features: InputFeatures):
        res = super().collate(features)
        strat_id = torch.tensor([f.labels[0] for f in features], dtype=torch.long) - len(self._tokenizer) + 8 

        if self._knowledge_name == 'basic':
            strat_id += 5
        elif self._knowledge_name == 'bm25':
            strat_id += 1
        elif self._knowledge_name == 'oracle':
            strat_id += 6
        elif self._knowledge_name in ['sbert','graph']:
            strat_id += 8   

        res['strat_id'] = strat_id
        
        return res

    def _norm(self, text: str):
        return ' '.join(text.strip().split())
    
    def _load_dataset_from_pkl(self, file_name):
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
            return dataset
        return None

    def _save_dataset_to_pkl(self, dataset, pkl_path):
        with open(pkl_path, 'wb') as f:
            pickle.dump(dataset, f)
