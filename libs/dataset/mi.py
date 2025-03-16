from transformers.tokenization_utils import PreTrainedTokenizer
from libs.dataset.base import BaseDataset


class MIDataset(BaseDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, stage: str) -> None:
        super().__init__(tokenizer, stage)

    def _convert_data_to_inputs(self, data: dict) -> list[dict]:
        process = lambda x: self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(x)
        )
        strat_id = process("[" + data["strategy"] + "]")
        assert len(strat_id) == 1
        strat_id = strat_id[0]
        knowledge = process(data["knowledge"]) + process(data["heal"])
        inputs = [
            {
                "context": [process(text) for text in data["dialog"]],
                "knowledge": knowledge,
                "response": process(data["target"]),
                "strat_id": strat_id,
            }
        ]

        self.inputs.append(
            {
                "context": [text for text in data["dialog"]],
                "response": self.tokenizer.decode(data["target"]),
            }
        )

        return inputs
