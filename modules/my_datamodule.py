import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
import lightning as L
from functools import partial
from torch.utils.data import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
from libs.config import Config
from libs.dataset import BaseDataset, MIDataset, ESConvDataset
from libs.utils.input_feature import InputFeature
from multiprocessing import cpu_count


class MyDataModule(L.LightningDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()
        self.BATCH_SIZE = Config.BATCH_SIZE
        self.tokenizer = tokenizer
        self.num_workers = min(4, cpu_count() // 2)

        MyDataset = BaseDataset
        if Config.DATA_NAME == 'esconv':
            MyDataset = ESConvDataset
        elif Config.DATA_NAME == 'mi':
            MyDataset = MIDataset

        self.train_dataset = MyDataset(tokenizer, stage="train")
        self.test_dataset = MyDataset(tokenizer, stage="test")
        self.dev_dataset = MyDataset(tokenizer, "valid")

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset.setup()
            self.dev_dataset.setup()

        if stage == "test" or stage == "predict":
            self.test_dataset.setup()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.BATCH_SIZE, 
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=partial(MyDataModule.collate, tokenizer=self.tokenizer)
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.BATCH_SIZE, 
            num_workers=self.num_workers,
            collate_fn=partial(MyDataModule.collate, tokenizer=self.tokenizer)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.BATCH_SIZE, 
            num_workers=self.num_workers,
            collate_fn=partial(MyDataModule.collate, tokenizer=self.tokenizer, is_test=True)
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.BATCH_SIZE, 
            num_workers=self.num_workers,
            collate_fn=partial(MyDataModule.collate, tokenizer=self.tokenizer, is_test=True)
        )
    
    @staticmethod
    def collate(features: List[InputFeature], tokenizer: PreTrainedTokenizer, is_test=False):
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        max_len = max([f.padding_length for f in features])
        
        input_ids = my_pad_sequence(
            [torch.tensor(f.input_ids, dtype=torch.long) for f in features],
            batch_first=True, 
            max_len=max_len, 
            padding_value=pad
        )
        attention_mask = my_pad_sequence(
            [torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
            batch_first=True, 
            max_len=max_len, 
            padding_value=0.
        )
        persona_input_ids = my_pad_sequence(
            [torch.tensor(f.persona_input_ids, dtype=torch.long) for f in features],
            batch_first=True, 
            max_len=max_len, 
            padding_value=pad
        )
        persona_attention_mask = my_pad_sequence(
            [torch.tensor([1.] * f.persona_input_length, dtype=torch.float) for f in features],
            batch_first=True,
            max_len=max_len, 
            padding_value=0.
        )
        labels = pad_sequence(
            [torch.tensor(f.labels, dtype=torch.long) for f in features],
            batch_first=True, 
            max_len=max_len,
            padding_value=-100
        )
        
        if not is_test:
            decoder_input_ids = pad_sequence(
                [torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                batch_first=True,
                padding_value=pad
            )
        else:
            decoder_input_ids = pad_sequence(
                torch.tensor([[f.decoder_input_ids[0]] for f in features], dtype=torch.long),
                batch_first=True,
                padding_value=pad
            )
        
        strat_id = torch.tensor([f.strat_id for f in features], dtype=torch.long) - len(tokenizer) + 8
        
        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            "persona_input_ids": persona_input_ids,
            "persona_attention_mask": persona_attention_mask,
            'labels': labels,
            'strat_id': strat_id,
        }
        
        return res


def my_pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0.0):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor
