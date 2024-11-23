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
            collate_fn=partial(
                    InputFeature.collate, 
                    tokenizer=self.tokenizer
                )
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.BATCH_SIZE, 
            num_workers=self.num_workers,
            collate_fn=partial(
                    InputFeature.collate, 
                    tokenizer=self.tokenizer
                )
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.BATCH_SIZE, 
            num_workers=self.num_workers,
            collate_fn=partial(
                    InputFeature.collate, 
                    tokenizer=self.tokenizer
                )
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.BATCH_SIZE, 
            num_workers=self.num_workers,
            collate_fn=partial(
                    InputFeature.collate, 
                    tokenizer=self.tokenizer
                )
        )
