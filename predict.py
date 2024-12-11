import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial

from modules.my_module import MyModule
from modules.my_datamodule import MyDataModule
from libs.dataset import BaseDataset, MIDataset, ESConvDataset
from libs.config import Config
from libs.utils.get_tokenizer import get_tokenizer
from libs.utils.get_model import get_model
from libs.utils.get_checkpoint import get_checkpoints

tokenizer = get_tokenizer()
# checkpoint = get_checkpoints()
# model = get_model("cpu", checkpoint)
model = get_model("cpu")
model.tie_tokenizer(tokenizer)

MyDataset = BaseDataset
if Config.DATA_NAME == 'esconv':
    MyDataset = ESConvDataset
elif Config.DATA_NAME == 'mi':
    MyDataset = MIDataset

dataset = MyDataset(tokenizer, "valid")
dataset.setup()


dataloader = DataLoader(
    dataset,
    batch_size=Config.BATCH_SIZE, 
    collate_fn=partial(MyDataModule.collate, tokenizer=tokenizer, is_test=True)
)

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predict Progress", total=len(dataloader)):
        encoded_info, generations = model.generate(**batch)
        breakpoint()


del dataset
del dataloader
