import lightning as L
from functools import partial
from torch.utils.data import DataLoader
from src.transformers.tokenization_utils import PreTrainedTokenizer
from libs.config import Config
from libs.dataset import BaseDataset, MIDataset, ESConvDataset
from libs.dataset.input_feature import InputFeature
from multiprocessing import cpu_count
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
from libs.config import Config


class MyDataModule(L.LightningDataModule):
    _dataset_esc: ESConvDataset = None

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__()

        self.BATCH_SIZE = Config.BATCH_SIZE
        self.tokenizer = tokenizer
        self.num_workers = min(4, cpu_count() // 2)

        MyDataset = BaseDataset
        if Config.DATA_NAME == "esconv":
            MyDataset = ESConvDataset
        elif Config.DATA_NAME == "mi":
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
            self.test_dataset.setup()
            MyDataModule._dataset_esc = self.test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.BATCH_SIZE,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=partial(MyDataModule.collate, tokenizer=self.tokenizer),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.BATCH_SIZE,
            num_workers=self.num_workers,
            collate_fn=partial(MyDataModule.collate, tokenizer=self.tokenizer),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.BATCH_SIZE,
            num_workers=self.num_workers,
            collate_fn=partial(MyDataModule.collate, tokenizer=self.tokenizer),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.BATCH_SIZE,
            num_workers=self.num_workers,
            collate_fn=partial(
                MyDataModule.collate, tokenizer=self.tokenizer, is_test=True
            ),
        )

    @classmethod
    def get_ref(cls, index: int) -> dict:
        if cls._dataset_esc is None:
            raise ValueError()
        return cls._dataset_esc[index]

    @staticmethod
    def collate(
        features: List[InputFeature], tokenizer: PreTrainedTokenizer, is_test=False
    ):
        pad = (
            tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )

        input_ids = pad_sequence(
            [torch.tensor(f.input_ids, dtype=torch.long) for f in features],
            batch_first=True,
            padding_value=pad,
        )
        attention_mask = pad_sequence(
            [torch.tensor([1.0] * f.input_length, dtype=torch.float) for f in features],
            batch_first=True,
            padding_value=0.0,
        )

        if is_test:
            decoder_input_ids = torch.tensor(
                [[f.decoder_input_ids[0]] for f in features], dtype=torch.long
            )
            labels = None
        else:
            decoder_input_ids = pad_sequence(
                [torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
                batch_first=True,
                padding_value=pad,
            )
            labels = pad_sequence(
                [torch.tensor(f.labels, dtype=torch.long) for f in features],
                batch_first=True,
                padding_value=-100,
            )

        res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }

        return res
