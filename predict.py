import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial

from modules.my_datamodule import MyDataModule
from libs.dataset import BaseDataset, MIDataset, ESConvDataset
from libs.config import Config
from libs.utils.get_tokenizer import get_tokenizer
from libs.utils.get_model import get_model
from libs.utils.get_checkpoint import get_checkpoints
from libs.utils.utils import cut_seq_to_eos, norm
from libs.metric.metrics import Metric

tokenizer = get_tokenizer()
# checkpoint = get_checkpoints()
# model = get_model("cpu", checkpoint)
model = get_model("cpu")
model.tie_tokenizer(tokenizer)

eos = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id

metric = Metric(tokenizer=tokenizer)
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

index = 0

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predict Progress", total=len(dataloader)):
        encoded_info, generations = model.generate(**batch)

        generations = [cut_seq_to_eos(each, eos) for each in generations.tolist()]

        for idx in range(len(generations)):
            r = norm(dataset.inputs[index]['response'])
            p = dataset.inputs[index]['context']
            ref, gen = [r], norm(tokenizer.decode(generations[idx]))

            metric.forword(ref, gen)
            
r_l = metric.calc_rouge_l
f1 = metric.calc_unigram_f1()
b_2 = metric.calc_bleu_k(k=2)
b_4 = metric.calc_bleu_k(k=4)
d_2 = metric.calc_distinct_k(k=2)
d_4 = metric.calc_distinct_k(k=4)

del dataset
del dataloader
