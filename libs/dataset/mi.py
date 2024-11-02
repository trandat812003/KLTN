from transformers.tokenization_utils import PreTrainedTokenizer
from libs.dataset.base import BaseDataset
from libs.config import Config


class MIDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(tokenizer)

    def _convert_data_to_inputs(self, data: dict):
        process = lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))

        strat_id = process('[' + data['strategy'] + ']')
        assert len(strat_id) == 1
        strat_id = strat_id[0]
        if Config.KNOWLEDGE_NAME == 'basic':
            knowledge = process(data['knowledge'])
        elif Config.KNOWLEDGE_NAME == 'bm25':
            knowledge = process('[knowledge]') + process(data['knowledge'])
        elif Config.KNOWLEDGE_NAME in ['sbert','graph']:
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