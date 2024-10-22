from typing import Any, List, Dict



class ESConvDataset:
    def __init__(self, data: Dict) -> None:
        self._data = data

    def convert_data_to_inputs(self, knowledge_name, process: Any) -> List:
        dialog = self._data['dialog']
        inputs = []
        context = []
        knowledge = []

        for i in range(len(dialog)):
            text = _norm(dialog[i]['text'])
            text = process(text)
            
            if dialog[i]['speaker'] == 'sys':
                strat_id = process('[' + dialog[i]['strategy'] + ']')
                assert len(strat_id) == 1
                strat_id = strat_id[0]
                
                if knowledge_name == 'oracle':
                    heal = process('[knowledge]') + process(dialog[i]['heal'])
                elif knowledge_name in ['sbert','graph']:
                    heal = process(dialog[i]['heal'])
                else:
                    heal = []
                    
            else:
                if knowledge_name in ['basic','oracle','sbert','graph']:
                    knowledge = process(dialog[i]['knowledge'])
                elif knowledge_name == 'bm25':
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

            context = context + [text]

        return inputs