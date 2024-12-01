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

        dialog = data['dialog']
        inputs = []
        context = []
        knowledge = []
        user_number = 0
        persona_list = data['persona_list']
        
        for i in range(len(dialog)):
            text = self._norm(dialog[i]['text'])
            if dialog[i]['speaker'] != 'sys':
                text = "Persona:" + text
                user_number += 1

        for i, turn in enumerate(dialog):
            text = process(self._norm(turn['text']))

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
                if Config.KNOWLEDGE_NAME in ['basic','oracle','sbert','graph']:
                    knowledge = process(dialog[i]['knowledge'])
                elif Config.KNOWLEDGE_NAME == 'bm25':
                    knowledge = process('[knowledge]') + process(dialog[i]['knowledge'])
                else:
                    knowledge = []
            
            if i > 0 and dialog[i]['speaker'] == 'sys':
                if user_number > 2:
                    index = user_number - 3
                    if index >= len(persona_list):
                        persona = persona_list[-1]
                    else:
                        persona = persona_list[index]
                else:
                    persona = "<input>"

                persona = persona.replace('<input>', '')
                persona = persona.replace('<persona>', '').strip()
                persona = "Persona Information: " + persona + "Dialogue: "

                persona = process(persona)

                res = {
                    'context': context.copy() + [process("System:")],
                    'knowledge': knowledge + heal,
                    'response': text,
                    'strat_id': strat_id,
                }

                inputs.append(res)

            context = context + [text]

        return inputs

    def _norm(self, s: str) -> str:
        return ' '.join(s.strip().split())
