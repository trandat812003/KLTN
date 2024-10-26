import os
from dotenv import load_dotenv
import pickle
import lightning as L
from torch.utils.data import DataLoader

from libs.dataset.esconv_dataset import ESConvDataset
from libs.dataset.mi_dataset import MIDataset


load_dotenv('./config/.env')


class MyDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self._batch_size = 1
        data_name = os.getenv("DATA_NAME")
        knowledge_name = os.getenv('KNOWLEDGE_NAME')

        if 'esc' in data_name:
            self._train_dataset = ESConvDataset(knowledge_name) 
            self._test_dataset = ESConvDataset(knowledge_name) 
            self._val_dataset = ESConvDataset(knowledge_name) 
        elif 'mi' in data_name:
            self._train_dataset = MIDataset(knowledge_name) 
            self._test_dataset = MIDataset(knowledge_name) 
            self._val_dataset = MIDataset(knowledge_name) 

    def setup(self, stage: str):
        if stage == "fit":
            train_data_path = os.path.join('.cache', self.data_dir, os.getenv("DATA_NAME"), "train.pkl")
            self._train_dataset = self._train_dataset.setup_and_load_dataset(self._train_dataset, train_data_path, "train")

            val_data_path = os.path.join('.cache', self.data_dir, os.getenv("DATA_NAME"), "valid.pkl")
            self._val_dataset = self._val_dataset.setup_and_load_dataset(self._val_dataset, val_data_path, "valid")

        elif stage == "test" or stage == "predict":
            test_data_path = os.path.join('.cache', self.data_dir, os.getenv("DATA_NAME"), "test.pkl")
            self._test_dataset = self._test_dataset.setup_and_load_dataset(self._test_dataset, test_data_path, "test")

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._batch_size)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self._batch_size)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self._batch_size)

    def predict_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self._batch_size)


__all__ = [
    "MyDataModule",
]