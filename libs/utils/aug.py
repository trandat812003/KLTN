import os
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from libs.dataset.augesc import AugDataset
from libs.config import Config
from modules.my_datamodule import MyDataModule
from tqdm import tqdm


def aug(tokenizer, model):
    aug_dataset = AugDataset(tokenizer, stage="augesc")
    aug_dataset.setup()

    aug_dataloader = DataLoader(
        aug_dataset,
        batch_size=Config.BATCH_SIZE, 
        collate_fn=partial(MyDataModule.collate, tokenizer=tokenizer)
    )

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "ln", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6)

    model.train()

    for batch in tqdm(aug_dataloader, desc="Augment Progress", total=len(aug_dataloader)):
        optimizer.zero_grad()
        labels = batch.pop("labels")

        outputs = model.aug(**batch)

        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), reduction="none")
        loss = loss.view(labels.size(0), labels.size(1))
        label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
        masked_lm_loss = torch.sum(loss) / torch.sum(label_size)

        loss = masked_lm_loss / (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS / batch["input_ids"].shape[0])
        loss.backward()
        optimizer.step()

    checkpoint_path = "./augmented_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    del aug_dataloader
    del aug_dataset


