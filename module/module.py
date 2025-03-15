import torch
import lightning as L
from src.transformers.tokenization_utils import PreTrainedTokenizer

# from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from libs.optimizer_moonlight import Moonlight
from libs.config import Config
from libs.my_logging import custom_logging
from module.model import MyModel
from libs.metrics import Metric, get_ppl_value


class MyModule(L.LightningModule):
    def __init__(self, tokenizer: PreTrainedTokenizer, model: MyModel):
        super().__init__()

        print(self.device)

        self.tokenizer = tokenizer
        self.model = model
        self.model.tie_tokenizer(self.tokenizer)

        self.save_hyperparameters(ignore=["model"])

        self.metrics = {"train": [], "val": [], "test": []}

        self.metric = Metric(tokenizer=tokenizer)

    def step(self, batch, batch_idx, phase):
        labels = batch["labels"]

        outputs = self.model(**batch)

        loss = outputs[0]
        logits = outputs[1]

        if phase == "train":
            label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
            loss = loss / torch.sum(label_size)
        else:
            labels[:, 0] = -100
            logits = logits[..., : self.tokenizer.vocab_size].contiguous()

        ppl_value, _ = get_ppl_value(logits, labels)
        self.metrics[phase].append(ppl_value)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "train")
        custom_logging.log({"train_epoch_loss": loss})

        current_lr = self.optimizers().param_groups[0]["lr"]
        custom_logging.log({"learning_rate": current_lr})
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
        ppl = torch.Tensor(self.metrics[phase]).mean()

        custom_logging.log({f"{phase}_ppl": ppl.item()})
        kwargs = {"epoch": self.current_epoch, "phase": phase, "ppl": ppl.item()}
        custom_logging.log_csv(**kwargs)

        self.metrics[phase] = []

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "ln", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        #     {"params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        # ]

        # optimizer = AdamW(optimizer_grouped_parameters, lr=Config.lr)
        # num_optim_steps = 12300 * Config.NUM_EPOCHS // Config.BATCH_SIZE + 1 # len(data_train) = 12285
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=100, num_training_steps=num_optim_steps
        # )

        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        muon_params = [
            p
            for name, p in param_optimizer
            if p.ndim >= 2
            and "embed_tokens" not in name
            and "lm_head" not in name
            and not any(nd in name for nd in no_decay)
        ]

        adamW_params = [
            p
            for name, p in param_optimizer
            if p.ndim < 2 and not any(nd in name for nd in no_decay)
        ]

        optimizer = Moonlight(
            lr=Config.lr,
            muon_params=muon_params,
            adamw_params=adamW_params,
        )

        return optimizer
