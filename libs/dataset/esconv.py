from transformers.tokenization_utils import PreTrainedTokenizer
from libs.dataset.base import BaseDataset
from libs.config import Config
from libs.utils.utils import norm


class ESConvDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, stage: str) -> None:
        super().__init__(tokenizer, stage)

    def _convert_data_to_inputs(self, data: dict) -> list[dict]:
        process = lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))
        dialog = data.get('dialog', [])
        if not dialog:
            raise ValueError("Dialog data is empty or missing.")

        inputs, context, knowledge = [], [], []
        user_number = 0
        persona_list = data['persona_list']

        for i, turn in enumerate(dialog):
            if turn['speaker'] == 'sys':
                strat_id = process('[' + turn['strategy'] + ']')
                assert len(strat_id) == 1, "Strategy ID must be a single token."
                strat_id = strat_id[0]

                heal = []
                if Config.KNOWLEDGE_NAME in ['sbert', 'graph']:
                    heal = process(turn['heal'])
            else:
                knowledge = process(turn['knowledge'])

            text = process(norm(turn['text']))
            if turn['speaker'] != 'sys':
                text = process(norm("Persona:" + turn['text']))
                user_number += 1
            else:
                text = process("System:") + [strat_id] + text
            
            if i > 0 and turn['speaker'] == 'sys' and (self.stage != 'test' or dialog[i - 1]['speaker'] != 'sys'):
                if user_number > 2:
                    index = user_number - 3
                    if index >= len(persona_list):
                        persona = persona_list[-1]
                    else:
                        persona = persona_list[index]
                else:
                    persona = "<input>"

                persona = process(persona)

                inputs.append({
                    'context': context.copy() + [process("System:")],
                    'knowledge': knowledge + heal,
                    'response': text,
                    'persona': persona,
                })

                self.inputs.append({
                    'context': [self.tokenizer.decode(c) for c in context.copy()],
                    'response': self.tokenizer.decode(text),
                })

            context = context + [text]

        return inputs
