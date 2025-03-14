from transformers.tokenization_utils import PreTrainedTokenizer
from libs.dataset.base import BaseDataset
from libs.utils import norm, weight_text_by_hybrid, weight_text_by_tf_ids
from libs.config import BlenderbotConfig


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
            text = process(norm(turn['text']))

            if turn['speaker'] == 'sys':
                strat_id = process('[' + turn['strategy'] + ']')
                assert len(strat_id) == 1, "Strategy ID must be a single token."
                words = norm(turn["text"]).split()
                vad_scores = [self.get_vad_scores(word) for word in words]
                weights = weight_text_by_hybrid(words, vad_scores[:, 1])
                v,a,d = self.compute_weighted_vad(vad_scores, weights)

                strat_ids = BlenderbotConfig.select_strategy(v,a,d)
                strat_ids = [process(text) for text in strat_ids] + [strat_id]

                heal = process(turn['heal'])
            else:
                knowledge = process(turn['knowledge'])
                

            if i > 0 and turn['speaker'] == 'sys':
                inputs.append({
                    'context': context.copy(),
                    'knowledge': knowledge + heal,
                    'response': text,
                    'strat_id': strat_ids,
                })

                self.inputs.append({
                    'context': [self.tokenizer.decode(c) for c in context.copy()],
                    'response': self.tokenizer.decode(text),
                })
            context = context + [text]

        return inputs
