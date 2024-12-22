from transformers.tokenization_utils import PreTrainedTokenizer
from libs.dataset.base import BaseDataset
from libs.config import Config
from libs.utils.utils import _norm


class ESConvDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, stage: str) -> None:
        super().__init__(tokenizer, stage)

    def _convert_data_to_inputs(self, data: dict) -> list[dict]:
        process = lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))
        dialog = data.get('dialog', [])
        if not dialog:
            raise ValueError("Dialog data is empty or missing.")

        inputs, context, knowledge = [], [], []

        for i, turn in enumerate(dialog):
            text = process(_norm(turn['text']))

            if turn['speaker'] == 'sys':
                strat_id = process('[' + turn['strategy'] + ']')
                assert len(strat_id) == 1, "Strategy ID must be a single token."
                strat_id = strat_id[0]

                heal = []
                if Config.KNOWLEDGE_NAME in ['sbert', 'graph']:
                    heal = process(turn['heal'])

            elif Config.KNOWLEDGE_NAME in ['bm25', 'oracle']:
                knowledge = process('[knowledge]') + process(turn['knowledge'])
            else:
                knowledge = process(turn['knowledge'])

            if i > 0 and turn['speaker'] == 'sys':
                inputs.append({
                    'context': context.copy(),
                    'knowledge': knowledge + heal,
                    'response': text,
                    'strat_id': strat_id,
                })

                self.inputs.append({
                    'context': [self.tokenizer.decode(c) for c in context.copy()],
                    'response': self.tokenizer.decode(text),
                })
            context = context + [text]

        return inputs
