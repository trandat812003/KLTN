import csv
import torch
from tqdm import tqdm

from libs.utils.get_model import get_model
from libs.utils.get_tokenizer import get_tokenizer
from modules.my_datamodule import MyDataModule
from libs.utils.utils import cut_seq_to_eos, norm


tokenizer = get_tokenizer()
datamodule = MyDataModule(tokenizer=tokenizer)
datamodule.setup(stage="predict")
model = get_model("cpu")
model.tie_tokenizer(tokenizer)
eos = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.sep_token_id
dataloader = datamodule.predict_dataloader()
dataset = datamodule.test_dataset

csv_filename = "predictions.csv"

with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # writer.writerow(["Reference", "Generated"])
    index = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predict Progress", total=len(dataloader)):
            batch.pop("labels")
            encoded_info, generations = model.generate(**batch)

            generations = [cut_seq_to_eos(each, eos) for each in generations.tolist()]

            for idx in range(len(generations)):
                r = norm(dataset.inputs[index]['response'])
                gen = norm(tokenizer.decode(generations[idx]))

                writer.writerow([f'Reference: "{r}"'])
                writer.writerow([f'Generated: "{gen}"'])
                writer.writerow([])
                index += 1

del dataset
del dataloader
