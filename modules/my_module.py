import torch
import numpy as np
import lightning as L
import torch.nn.functional as F
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from libs.utils.get_model import get_model
from libs.utils.get_tokenizer import get_tokenizer
from libs.config import Config


class MyModule(L.LightningModule):
    def __init__(self):
        super().__init__()

        self._tokenizer = get_tokenizer()
        self._model = get_model()
        self._model.tie_tokenizer(self._tokenizer)

        self.save_hyperparameters()

        self._model.to(self.device)

        print(self.device)

        self.metrics = {
            "train": {"loss": 0.0, "steps": 0},
            "val": {"loss": 0.0, "steps": 0},
            "test": {"loss": 0.0, "steps": 0}
        }

    def step(self, batch, batch_idx, phase):
        labels = batch.pop("labels")
        strat_id = batch.pop("strat_id", None)

        outputs = self._model(**batch)
        if phase == "val":
            labels[:, 0] = -100
            outputs = outputs[..., :self._tokenizer.vocab_size].contiguous()

        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), reduction="none")
        loss = loss.view(labels.size(0), labels.size(1))
        label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
        masked_lm_loss = torch.sum(loss) / torch.sum(label_size)

        input_ids = batch["input_ids"]
        tmp_loss = float(torch.sum(loss).item())

        metrics = self.metrics[phase]
        metrics["loss"] += tmp_loss
        metrics["steps"] += label_size.sum().float()

        return masked_lm_loss / (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS / input_ids.shape[0])
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def on_train_epoch_end(self):
        self.on_epoch_end("train")

    def on_validation_epoch_end(self):
        self.on_epoch_end("val")

    def on_test_epoch_end(self):
        self.on_epoch_end("test")

    def on_epoch_end(self, phase):
        metrics = self.metrics[phase]
        loss = metrics["loss"] / metrics["steps"]
        ppl = np.exp(loss)

        self.log(f"{phase}_loss", loss, prog_bar=True)
        self.log(f"{phase}_ppl", ppl, prog_bar=True)

        metrics["loss"], metrics["steps"] = 0.0, 0

    def configure_optimizers(self):
        param_optimizer = list(self._model.named_parameters())
        no_decay = ["bias", "ln", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
        num_optim_steps = 2455
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=num_optim_steps
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
