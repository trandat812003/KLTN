import torch
import csv
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from functools import partial
from lightning import Trainer

from modules.my_module import MyModule
from modules.my_datamodule import MyDataModule
from libs.config import Config
from libs.utils.get_tokenizer import get_tokenizer
from libs.utils.get_model import get_model
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


#################################################################
# predict

model = module.model
eos = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id

metric = Metric(tokenizer=tokenizer)
datamodule.setup("test")
dataloader = datamodule.test_dataloader()

test_loss = 0
test_steps = 0
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Test Progress", total=len(dataloader)):
        labels = batch.pop("labels")

        outputs = model(**batch)

        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), reduction="none")
        loss = loss.view(labels.size(0), labels.size(1))
        label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
        masked_lm_loss = torch.sum(loss) / torch.sum(label_size)

        input_ids = batch["input_ids"]
        tmp_loss = float(torch.sum(loss).item())

        test_loss += tmp_loss
        test_steps += label_size.sum().float()

with open(logging.csv_path, "a", newline="") as file:
    writer = csv.writer(file)
    loss = test_loss / test_steps
    ppl = np.exp(loss)
    writer.writerow(["TEST"])
    writer.writerow(["loss", "ppl"])
    writer.writerow([loss, ppl])

dataloader = datamodule.predict_dataloader()
dataset = datamodule.test_dataset

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
            
r_l = metric.calc_rouge_l()
f1 = metric.calc_unigram_f1()
b_2 = metric.calc_bleu_k(k=2)
b_4 = metric.calc_bleu_k(k=4)
d_2 = metric.calc_distinct_k(k=2)
d_4 = metric.calc_distinct_k(k=4)

with open(logging.csv_path, "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["PREDICT"])
    writer.writerow(["R_L", "f1", "b_2", "b_4", "d_2", "d_4"])
    writer.writerow([r_l, f1, b_2, b_4, d_2, d_4])

del dataset
del dataloader
