import json
from libs.dataset.my_dataset import MyDataset
from libs.models.autotokenizer import get_tokenizer
from libs.utils import InputFeatures


class MIDataset(MyDataset):
    def __init__(self, knowledge_name: str) -> None:
        super().__init__(knowledge_name)

    def setup(self):
        self._tokenizer = get_tokenizer("esconv", self._knowledge_name)

        train_data_path = f"./dataset/mi/{self._knowledge_name}/train.txt"
        with open(train_data_path,'r',encoding='utf-8') as f:
            reader = f.readlines()

        for line in reader:
            data = json.loads(line)

            inputs = self.convert_data_to_inputs(data)
            self._data.append(self.convert_inputs_to_feature(inputs))

    def convert_data_to_inputs(self, data: dict) -> list:
        process = lambda x: self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(x))

        strat_id = process('[' + data['strategy'] + ']')
        assert len(strat_id) == 1
        strat_id = strat_id[0]
        if self._knowledge_name == 'basic':
            knowledge = process(data['knowledge'])
        elif self._knowledge_name == 'bm25':
            knowledge = process('[knowledge]')+process(data['knowledge'])
        elif self._knowledge_name in ['sbert','graph']:
            knowledge = process(data['knowledge']) + process(data['heal'])
        else:
            knowledge = []
        inputs = [
            {
                'context': [process(text) for text in data['dialog']], 
                'knowledge': knowledge,
                'response': process(data['target']), 
                'strat_id': strat_id
            }
        ]

        return inputs
    
    def collate(self, feature: InputFeatures):
        res = super().collate(feature)
        strat_id = feature.labels[0] - len(self._tokenizer) + 10

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
    
    