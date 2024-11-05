
import torch
import lightning as L
import torch.nn.functional as F
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import Seq2SeqLMOutput

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

        self.tr_loss = 0.0
        self.tr_ppl = 0.0
        self.nb_tr_examples = 0
        self.nb_tr_steps = 0

    def training_step(self, batch, batch_idx):
        breakpoint()
        labels = batch.pop('labels')
        strat_id = batch.pop('strat_id')
        outputs = self._model(**batch)
        loss, ppl = self._calculator_loss_and_ppl_value(outputs, labels)

        input_ids = batch['input_ids']

        tmp_loss = float(loss.item()) * (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS / input_ids.shape[0])
        self.tr_loss += tmp_loss
        self.nb_tr_examples += input_ids.size(0)
        self.nb_tr_steps += 1
        
        tmp_ppl = ppl.item() if ppl.item() < float('inf') else self.tr_ppl
        self.tr_ppl += tmp_ppl

        return loss
    
    def on_train_epoch_end(self):
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self.log("train_loss", self.tr_loss / self.nb_tr_steps, prog_bar=True)
        self.log("train_ppl", self.tr_ppl / self.nb_tr_steps, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        strat_id = batch.pop('strat_id')
        outputs = self._model(**batch)
        labels[:, 0] = -100
        outputs = outputs[..., :self._tokenizer.vocab_size].contiguous()
        loss, ppl = self._calculator_loss_and_ppl_value(outputs, labels)

        print(f"Step {batch_idx + 1} - Loss: {loss:.4f}, PPL: {ppl:.4f}")

        return loss
    
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        strat_id = batch.pop('strat_id')
        outputs = self._model(**batch)

        loss, ppl = self._calculator_loss_and_ppl_value(outputs, labels)

        print(f"Step {batch_idx + 1} - Loss: {loss:.4f}, PPL: {ppl:.4f}")

        return loss

    def _calculator_loss_and_ppl_value(self, predict, labels):
        loss = F.cross_entropy(predict.view(-1, predict.size(-1)), labels.view(-1), reduction='none')
        loss = loss.view(labels.size(0), labels.size(1))
        label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
        ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))
        loss = torch.sum(loss) / torch.sum(label_size)

        return loss, ppl_value

    def configure_optimizers(self):
        param_optimizer = list(self._model.named_parameters())
        no_decay = ['bias', 'ln', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,)

        # num_optim_steps = os.getenv('NUM_EPOCHS') * (len(train_dataloader) // 
        # args.train_batch_size + int(len(train_dataloader) % args.train_batch_size != 0))
        
        num_optim_steps = 2455
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=num_optim_steps
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
