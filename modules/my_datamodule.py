import lightning as L
from functools import partial
from torch.utils.data import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
from libs.config import Config
from libs.dataset import BaseDataset, MIDataset, ESConvDataset
from libs.utils.input_feature import InputFeature
from multiprocessing import cpu_count
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
from libs.config import Config


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

        if stage == "test":
            self.test_dataset.setup()

        if stage == "predict":
            self.dev_dataset.setup()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.BATCH_SIZE, 
            num_workers=self.num_workers,
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
            collate_fn=partial(MyDataModule.collate, tokenizer=self.tokenizer)
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.BATCH_SIZE, 
            num_workers=self.num_workers,
            collate_fn=partial(MyDataModule.collate, tokenizer=self.tokenizer, is_test=True)
        )
    
    @staticmethod
    def collate(features: List[InputFeature], tokenizer: PreTrainedTokenizer, is_test=False):
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        
        input_ids = pad_sequence(
            [torch.tensor(f.input_ids, dtype=torch.long) for f in features],
            batch_first=True, 
            padding_value=pad
        )
        attention_mask = pad_sequence(
            [torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
            batch_first=True, 
            padding_value=0.
        )
        labels = pad_sequence(
            [torch.tensor(f.labels, dtype=torch.long) for f in features],
            batch_first=True, 
            padding_value=-100
        )
        
        if is_test:
            decoder_input_ids = torch.tensor([[f.decoder_input_ids[0]] for f in features], dtype=torch.long)
        else:
            decoder_input_ids = pad_sequence(
                [torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                batch_first=True, 
                padding_value=pad
            )
            
        
        if Config.DATA_NAME == 'esconv':
            strat_id = torch.tensor([f.labels[0] for f in features], dtype=torch.long) - len(tokenizer) + 8
        elif Config.DATA_NAME == 'mi':
            strat_id = torch.tensor([f.labels[0] for f in features], dtype=torch.long) - len(tokenizer) + 10
        
        if Config.KNOWLEDGE_NAME == 'basic':
            strat_id += 5
        elif Config.KNOWLEDGE_NAME == 'bm25':
            strat_id += 1
        elif Config.KNOWLEDGE_NAME == 'oracle':
            strat_id += 6
        elif Config.KNOWLEDGE_NAME in ['sbert','graph']:
            strat_id += 8
        
        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
            'strat_id': strat_id,
        }
        
        return res
