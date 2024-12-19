import torch
import numpy as np
import lightning as L
import torch.nn.functional as F
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from libs.config import Config, Logging

logging = Logging()

class MyModule(L.LightningModule):
    def __init__(self, tokenizer: PreTrainedTokenizer, model):
        super().__init__()

        print(self.device)

        self.tokenizer = tokenizer
        self.model = model
        self.model.tie_tokenizer(self.tokenizer)

        self.save_hyperparameters()

        self.model.to(self.device)

        self.metrics = {
            "train": {"loss": 0.0, "steps": 0},
            "val": {"loss": 0.0, "steps": 0},
            "test": {"loss": 0.0, "steps": 0}
        }

    def step(self, batch, batch_idx, phase):
        labels = batch.pop("labels")

        outputs = self.model(**batch)
        if phase == "val":
            labels[:, 0] = -100
            outputs = outputs[..., :self.tokenizer.vocab_size].contiguous()

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
        loss = self.step(batch, batch_idx, "train")
        logging.log({"train_epoch_loss": self.metrics['train']["loss"] / self.metrics['train']["steps"]})

        current_lr = self.optimizers().param_groups[0]["lr"]
        logging.log({"learning_rate": current_lr})
        return loss

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

        logging.log({f"{phase}_loss": loss, f"{phase}_ppl": ppl})
        logging.log_csv(self.current_epoch, phase, loss, ppl)

        metrics["loss"], metrics["steps"] = 0.0, 0

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "ln", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
        num_optim_steps = 12300 * Config.NUM_EPOCHS // Config.BATCH_SIZE + 1 # len(data_train) = 12285
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=1000, num_training_steps=num_optim_steps
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
