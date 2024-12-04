from transformers.tokenization_utils import PreTrainedTokenizer
from libs.dataset.base import BaseDataset
from libs.config import Config


class ESConvDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, stage: str) -> None:
        super().__init__(tokenizer, stage)

    def _convert_data_to_inputs(self, data: dict) -> list[dict]:
        dialog = data.get('dialog', [])
        if not dialog:
            raise ValueError("Dialog data is empty or missing.")

        dialog = data['dialog']
        inputs = []
        context = []
        knowledge = ''
        user_number = 0
        persona_list = data['persona_list']

        for i, turn in enumerate(dialog):
            text = self._norm(dialog[i]['text'])
            if dialog[i]['speaker'] != 'sys':
                text = "Persona:" + text
                user_number += 1

            if turn['speaker'] == 'sys':
                strat_id = '[' + turn['strategy'] + ']'

                heal = ''
                if Config.KNOWLEDGE_NAME in ['sbert', 'graph']:
                    heal = turn['heal']

            elif Config.KNOWLEDGE_NAME in ['bm25', 'oracle']:
                knowledge = '[knowledge]' + turn['knowledge']
            else:
                if Config.KNOWLEDGE_NAME in ['basic','oracle','sbert','graph']:
                    knowledge = dialog[i]['knowledge']
                elif Config.KNOWLEDGE_NAME == 'bm25':
                    knowledge = '[knowledge]' + dialog[i]['knowledge']
                else:
                    knowledge = ''
            
            if i > 0 and dialog[i]['speaker'] == 'sys' and (self.stage != 'test' or dialog[i - 1]['speaker'] != 'sys'):
                last_text = self._norm(dialog[i - 1]['text'])
                if user_number > 2:
                    index = user_number - 3
                    if index >= len(persona_list):
                        persona = persona_list[-1]
                    else:
                        persona = persona_list[index]
                else:
                    persona = "<input>"

                persona = persona.replace('<input>', '')
                persona = persona.replace('<persona>', '', 1).strip()
                persona = "</s> <s>".join([p.strip() for p in persona.split('<persona>')])
                persona = "Persona Information:\n" + persona + '</s> <s>' + "Dialogue:\n"

                res = {
                    'last_text': last_text,
                    'context': context.copy() + ["System:"],
                    'knowledge': knowledge + heal,
                    'strat_id': strat_id,
                    'persona': persona
                }

                inputs.append(res)

            if dialog[i]['speaker'] == 'sys':
                text = "System:" + strat_id + text

            context = context + [text]

        return inputs

    def _norm(self, s: str) -> str:
        return ' '.join(s.strip().split())
