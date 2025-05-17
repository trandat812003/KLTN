import torch
import lightning as L
from transformers.tokenization_utils import PreTrainedTokenizer

# from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.optim import AdamW

# from libs.optimizer_moonlight import Moonlight
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
        
        if Config.CHECKPOINT_PATH:
            print(f"Loading checkpoint from {Config.CHECKPOINT_PATH}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=device)

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    print("Detected PyTorch Lightning checkpoint.")
                else:
                    state_dict = checkpoint
                    print("Detected standard PyTorch checkpoint.")
            else:
                raise ValueError("Unsupported checkpoint format.")

            self.model.load_state_dict(state_dict, strict=False)

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
            
        logits_cpu = logits.detach().cpu()
        labels_cpu = labels.detach().cpu()
        ppl_value, _ = get_ppl_value(logits_cpu, labels_cpu)

        self.metrics[phase].append(ppl_value)

        del batch, labels, logits, outputs, logits_cpu, labels_cpu
        torch.cuda.empty_cache()

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
    
        ckpt_dir = os.path.join(Config.OUTPUT_DIR, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        print("✅ Saved checkpoint")


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
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=Config.lr)
        # num_optim_steps = 12300 * Config.NUM_EPOCHS // Config.BATCH_SIZE + 1 # len(data_train) = 12285
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=100, num_training_steps=num_optim_steps
        # )

        # return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer
