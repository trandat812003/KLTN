from transformers.tokenization_utils import PreTrainedTokenizer
from libs.dataset.base import BaseDataset
from libs.config import Config


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
            text = self._norm(turn.get('text', ""))
            text_tokens = process(text)

            if turn['speaker'] == 'sys':
                strat_id = process('[' + turn['strategy'] + ']')
                assert len(strat_id) == 1, "Strategy ID must be a single token."
                strat_id = strat_id[0]

                heal = []
                if Config.KNOWLEDGE_NAME in ['sbert', 'graph']:
                    heal = process(turn['heal'])

            elif Config.KNOWLEDGE_NAME in ['bm25', 'oracle']:
                knowledge = process('[knowledge]') + process(turn.get('knowledge', ""))
            else:
                knowledge = process(turn.get('knowledge', ""))

            if i > 0 and turn['speaker'] == 'sys':
                inputs.append({
                    'context': context + [process("System:")],
                    'knowledge': knowledge + heal,
                    'response': text_tokens,
                    'strat_id': strat_id,
                })

            context.append(text_tokens)

        return inputs

    def _norm(self, s: str) -> str:
        return ' '.join(s.strip().split())
