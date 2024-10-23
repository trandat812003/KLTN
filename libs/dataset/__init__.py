import os
import pickle
import lightning as L
from torch.utils.data import DataLoader

from libs.dataset.my_dataset import MyDataset
from libs.dataset.esconv_dataset import ESConvDataset
from libs.dataset.mi_dataset import MIDataset


class MyDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./", knowledge_name: str = "sbert"):
        super().__init__()
        self.data_dir = data_dir
        self._batch_size = 4

        if 'esc' in self.data_dir:
            self.train_dataset = ESConvDataset(knowledge_name) 
            self.test_dataset = ESConvDataset(knowledge_name) 
            self.val_dataset = ESConvDataset(knowledge_name) 
        elif 'mi' in self.data_dir:
            self.train_dataset = MIDataset(knowledge_name) 
            self.test_dataset = MIDataset(knowledge_name) 
            self.val_dataset = MIDataset(knowledge_name) 

    def setup(self, stage: str):
        def load_dataset_from_pkl(file_name):
            if os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    dataset = pickle.load(f)
                return dataset
            return None
        
        def save_dataset_to_pkl(dataset, file_name):
            pkl_path = os.path.join(self.data_dir, file_name)
            with open(pkl_path, 'wb') as f:
                pickle.dump(dataset, f)

        if stage == "fit":
            train_data_path = os.path.join(self.data_dir, "train_dataset.pkl")
            if not os.path.exists(train_data_path):
                self.train_dataset.setup("train")
                save_dataset_to_pkl(self.train_dataset, train_data_path)
            else:
                self.train_dataset = load_dataset_from_pkl(train_data_path)

            val_data_path = os.path.join(self.data_dir, "val_dataset.pkl")
            if not os.path.exists(val_data_path):
                self.val_dataset.setup("valid")
                save_dataset_to_pkl(self.val_dataset, val_data_path)
            else:
                self.val_dataset = load_dataset_from_pkl(val_data_path)

        if stage == "test":
            test_data_path = os.path.join(self.data_dir, "test_dataset.pkl")
            if not os.path.exists(test_data_path):
                self.test_dataset.setup("test")
                save_dataset_to_pkl(self.test_dataset, test_data_path)
            else:
                self.test_dataset = load_dataset_from_pkl(test_data_path)

        if stage == "predict":
            if not os.path.exists(test_data_path):
                self.test_dataset.setup("test")
                save_dataset_to_pkl(self.test_dataset, test_data_path)
            else:
                self.test_dataset = load_dataset_from_pkl(test_data_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self._batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self._batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self._batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self._batch_size)
