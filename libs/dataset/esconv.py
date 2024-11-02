from transformers.tokenization_utils import PreTrainedTokenizer
from libs.dataset.base import BaseDataset
from libs.config import Config


class ESConvDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(tokenizer)

    def _convert_data_to_inputs(self, data: dict):
        process = lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))

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
                
                if Config.KNOWLEDGE_NAME == 'oracle':
                    heal = process('[knowledge]') + process(dialog[i]['heal'])
                elif Config.KNOWLEDGE_NAME in ['sbert','graph']:
                    heal = process(dialog[i]['heal'])
                else:
                    heal = []
                    
            else:
                if Config.KNOWLEDGE_NAME in ['basic','oracle','sbert','graph']:
                    knowledge = process(dialog[i]['knowledge'])
                elif Config.KNOWLEDGE_NAME == 'bm25':
                    knowledge = process('[knowledge]') + process(dialog[i]['knowledge'])
                else:
                    knowledge = []
            
            if i > 0 and dialog[i]['speaker'] == 'sys':
                
                res = {
                    'context': context.copy(),
                    'knowledge': knowledge + heal,
                    'response': text,
                    'strat_id': strat_id,
                }
                
                inputs.append(res)

            #if dialog[i]['speaker'] == 'sys':
            #    text = [strat_id] + text

            context = context + [text]

        return inputs
            
    def _norm(self, s: str):
        return ' '.join(s.strip().split())