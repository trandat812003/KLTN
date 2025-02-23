import csv
import torch
import numpy as np
import lightning as L
import torch.nn.functional as F
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from libs.utils.utils import cut_seq_to_eos, norm
from libs.config import Config, Logging
from modules.my_model import MyModel
from modules.my_datamodule import MyDataModule
from libs.metric.metrics import Metric

logging = Logging()

class MyModule(L.LightningModule):
    def __init__(self, tokenizer: PreTrainedTokenizer, model: MyModel):
        super().__init__()

        print(self.device)

        self.tokenizer = tokenizer
        self.model = model
        self.model.tie_tokenizer(self.tokenizer)

        self.save_hyperparameters(ignore=['model'])

        self.metrics = {
            "train": {"loss": 0.0, "steps": 0},
            "val": {"loss": 0.0, "steps": 0},
            "test": {"loss": 0.0, "steps": 0}
        }

        self.metric = Metric(tokenizer=tokenizer)

    def step(self, batch, batch_idx, phase):
        strat_id = batch.pop("strat_id")
        labels = batch["labels"]
        # input_ids = batch["input_ids"]

        outputs = self.model(**batch)
        if phase == "val" or phase == "train":
            outputs = outputs[1]

        if phase == "val":
            labels[:, 0] = -100
            outputs = outputs[..., :self.tokenizer.vocab_size].contiguous()
        else:
            alpha_l = []

            lm_size = outputs.size()
            
            for i in strat_id:
                tmp_alpha = self.model.strategy_alpha[i.item()]
                tmp_alpha = tmp_alpha * torch.ones(lm_size[1], lm_size[2], device=self.device)
                alpha_l.append(tmp_alpha)
            alpha_l = torch.stack(alpha_l)

            outputs = (torch.ones_like(outputs, device=self.device)+alpha_l)*outputs - alpha_l*outputs

        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), reduction='none')
        loss = torch.sum(loss.view(labels.size(0), labels.size(1)))
        label_size = torch.sum(torch.sum(labels.ne(-100), dim=1).type_as(loss))

        # breakpoint()

        metrics = self.metrics[phase]
        metrics["loss"] += (loss / label_size).item()
        metrics["steps"] += label_size

        return loss / label_size
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "train")
        logging.log({"train_epoch_loss": self.metrics['train']["loss"]/ self.metrics['train']["steps"]})

        current_lr = self.optimizers().param_groups[0]["lr"]
        logging.log({"learning_rate": current_lr})
        return loss

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx):
        strat_id = batch.pop("strat_id")
        eos = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else self.tokenizer.sep_token_id

        with torch.no_grad():
            encoded_info, generations = self.model.generate(**batch)

            generations = [cut_seq_to_eos(each, eos) for each in generations.tolist()]

            for idx in range(len(generations)):
                true_predict = MyDataModule.get_ref(batch_idx * Config.BATCH_SIZE + idx)
                r = norm(true_predict['response'])
                p = true_predict['context']
                ref, gen = [r], norm(self.tokenizer.decode(generations[idx]))

                self.metric.forword(ref, gen)

                logging.log_predict(ref=ref, gen=gen)

    def on_train_epoch_end(self):
        self.on_epoch_end("train")

    def on_validation_epoch_end(self):
        self.on_epoch_end("val")

    def on_test_epoch_end(self):
        self.on_epoch_end("test")

    def on_predict_epoch_end(self):
        r_l = self.metric.calc_rouge_l()
        f1 = self.metric.calc_unigram_f1()
        b_2 = self.metric.calc_bleu_k(k=2)
        b_4 = self.metric.calc_bleu_k(k=4)
        d_2 = self.metric.calc_distinct_k(k=2)
        d_4 = self.metric.calc_distinct_k(k=4)

        with open(logging.csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["PREDICT"])
            writer.writerow(["R_L", "f1", "b_2", "b_4", "d_2", "d_4"])
            writer.writerow([r_l, f1, b_2, b_4, d_2, d_4])

    def on_epoch_end(self, phase):
        metrics = self.metrics[phase]
        loss = metrics["loss"] / metrics["steps"]
        ppl = np.exp(loss.item())

        logging.log({f"{phase}_loss": loss.item(), f"{phase}_ppl": ppl})
        logging.log_csv(self.current_epoch, phase, loss.item(), ppl)

        metrics["loss"], metrics["steps"] = 0.0, 0

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "ln", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=Config.lr)
        num_optim_steps = 12300 * Config.NUM_EPOCHS // Config.BATCH_SIZE + 1 # len(data_train) = 12285
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=num_optim_steps
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        # return optimizer
