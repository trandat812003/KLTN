import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
from lightning import Trainer

from modules.my_module import MyModule
from modules.my_datamodule import MyDataModule
from libs.dataset import BaseDataset, MIDataset, ESConvDataset
from libs.config import Config
from libs.utils.get_tokenizer import get_tokenizer
from libs.utils.get_model import get_model
from libs.utils.get_checkpoint import get_checkpoints
from libs.utils.utils import cut_seq_to_eos, _norm
from libs.metric.metrics import Metric
from libs.config import Config, Logging

logging = Logging()

tokenizer = get_tokenizer()
datamodule = MyDataModule(tokenizer=tokenizer)
model = get_model("cpu")
module = MyModule(tokenizer, model)

trainer = Trainer(
    max_epochs=Config.NUM_EPOCHS, 
    gradient_clip_val=Config.GRADIENT_ACCUMULATION_STEPS,
    accelerator="cpu",
)
trainer.fit(module, datamodule=datamodule)
trainer.test(module, datamodule=datamodule)


#################################################################
# predict
# checkpoint = get_checkpoints()
# model = get_model("cpu", checkpoint)
# model.tie_tokenizer(tokenizer)

model = module.model
eos = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id

metric = Metric(tokenizer=tokenizer)
dataset = datamodule.test_dataset
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
            r = _norm(dataset.inputs[index]['response'])
            p = dataset.inputs[index]['context']
            ref, gen = [r], _norm(tokenizer.decode(generations[idx]))

            metric.forword(ref, gen)
            
r_l = metric.calc_rouge_l
f1 = metric.calc_unigram_f1()
b_2 = metric.calc_bleu_k(k=2)
b_4 = metric.calc_bleu_k(k=4)
d_2 = metric.calc_distinct_k(k=2)
d_4 = metric.calc_distinct_k(k=4)

logging.log({f"R_L": r_l})
logging.log({f"f1": f1})
logging.log({f"b_2": b_2})
logging.log({f"b_4": b_4})
logging.log({f"d_2": d_2})
logging.log({f"d_4": d_4})


del dataset
del dataloader
