from transformers.tokenization_utils import PreTrainedTokenizer
from libs.dataset.base import BaseDataset
from libs.config import Config


class MIDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, stage: str) -> None:
        super().__init__(tokenizer, stage)

    def _convert_data_to_inputs(self, data: dict) -> list[dict]:
        process = lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))

        strat_id = process('[' + data['strategy'] + ']')
        assert len(strat_id) == 1, "Strategy ID must be a single token."
        strat_id = strat_id[0]

        knowledge = self._process_knowledge(data, process)

        return [{
            'context': [process(utterance) for utterance in data['dialog']],
            'knowledge': knowledge,
            'response': process(data['target']),
            'strat_id': strat_id
        }]

    def _process_knowledge(self, data: dict, process) -> list[int]:
        if Config.KNOWLEDGE_NAME == 'bm25':
            return process('[knowledge]') + process(data['knowledge'])
        elif Config.KNOWLEDGE_NAME in ['sbert', 'graph']:
            return process(data['knowledge']) + process(data['heal'])
        return process(data['knowledge'])
