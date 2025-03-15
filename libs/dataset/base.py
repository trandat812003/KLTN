import json
from transformers.tokenization_utils import PreTrainedTokenizer
from torch.utils.data import Dataset
from libs.dataset.input_feature import InputFeature
from libs.config import Config
from libs.utils import save_file_pickle, load_file_pickle, read_file


class BaseDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, stage: str) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.stage = stage
        self.data_list = []
        self.inputs = []
        self.VAD = None

    def setup(self) -> None:
        self.data_list, self.inputs = load_file_pickle(
            root_file=f'./tmp/dataset/{Config.DATA_NAME}',
            file_name=f'{self.stage}.pkl'
        )

        if self.data_list is not None:
            return
        
        self.data_list = []
        self.inputs = []
        reader = read_file(f'./DATA/{Config.DATA_NAME}/{self.stage}.txt')

        for line in reader:
            data = json.loads(line)
            inputs = self._convert_data_to_inputs(data)
            features = self._convert_inputs_to_features(inputs)
            self.data_list.extend(features)

        save_file_pickle(
            f'./tmp/dataset/{Config.DATA_NAME}/{self.stage}.pkl',
            {'data_list': self.data_list, 'inputs': self.inputs}
        )

    def _convert_inputs_to_features(self, inputs: list[dict]) -> list[InputFeature]:
        if not inputs:
            return []
        
        features = [self._featurize(**ipt) for ipt in inputs]
        return features
    
    def _featurize(self, context: list[int], knowledge: list[int], response: list[int], strat_id: list[int]) -> InputFeature:
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else  self.tokenizer.cls_token_id
        eos = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else  self.tokenizer.sep_token_id

        if not all([pad, bos, eos]):
            raise ValueError("Token IDs (pad, bos, eos) must be defined.")

        context = [c + [eos] for c in context]
        context += [knowledge + [eos]]
        input_ids = sum(context, [])[-(Config.MAX_INPUT_LENGTH - len(strat_id)):]
        input_ids += strat_id[:-1]

        labels = (strat_id[-1] + response + [eos])[:Config.MAX_DECODER_INPUT_LENGTH + 1]
        decoder_input_ids = [bos] + labels[:-1]

        assert len(decoder_input_ids) == len(labels), "Mismatch between decoder inputs and labels"

        return InputFeature(input_ids, decoder_input_ids, labels)
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index: int):
        return self.data_list[index]
    
    def _convert_data_to_inputs(self, *args) -> list[dict]:
        raise NotImplementedError("This method should be implemented in subclasses.")
