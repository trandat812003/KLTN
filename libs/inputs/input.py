from transformers.tokenization_utils import PreTrainedTokenizer
from typing_extensions import List

from data import ESConvDataset, MIDataset


class Input:
    def __init__(self, data_name: str) -> None:
        self.data_name = data_name

    def convert_data_to_inputs(self, token: PreTrainedTokenizer) -> List:
        pass