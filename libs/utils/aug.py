import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import os

from libs.dataset.augment import AugmentDataset
from libs.config import Config
from module.data import MyDataModule


def aug(tokenizer, model):
    # Thiết bị huấn luyện
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dùng DataParallel để sử dụng nhiều GPU
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.train()

    # Dataset augment
    aug_dataset = AugmentDataset(tokenizer, stage="aug")
    aug_dataset.setup()

    aug_dataloader = DataLoader(
        aug_dataset,
        batch_size=Config.BATCH_SIZE,
        collate_fn=partial(MyDataModule.collate, tokenizer=tokenizer),
    )

    # Cấu hình optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "ln", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-6)

    # Gắn tokenizer
    model.module.tie_tokenizer(tokenizer)

    # Vòng lặp train
    for batch in tqdm(aug_dataloader, desc="Augment Progress", total=len(aug_dataloader)):
        optimizer.zero_grad()
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model.module.aug(**batch)

        loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            labels.view(-1),
            reduction="none"
        )

        loss = loss.view(labels.size(0), labels.size(1))
        label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
        masked_lm_loss = torch.sum(loss) / torch.sum(label_size)

        loss = masked_lm_loss / (
            Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS / batch["input_ids"].shape[0]
        )

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

    # 📦 Lưu checkpoint
    checkpoint_dir = os.path.join(Config.OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "augmented_model.pt")
    torch.save(model.module.state_dict(), checkpoint_path)
    print(f"✅ Checkpoint saved to: {checkpoint_path}")

    # Giải phóng
    del aug_dataloader
    del aug_dataset

    return model.module  # Trả về model không bọc DataParallel
