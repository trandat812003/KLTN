
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

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        strat_id = batch.pop('strat_id')
        outputs = self._model(**batch)
        loss = self._calculator_loss(outputs, labels)
        ppl = self._calculator_ppl_value(outputs, labels)

        input_ids = batch['input_ids']

        tmp_loss = float(loss.item()) * (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS/ input_ids.shape[0])
        tr_loss += tmp_loss
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        mean_loss = tr_loss / nb_tr_steps

        print(mean_loss)

        return loss
    
    def on_train_epoch_end(self):
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
    
    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        strat_id = batch.pop('strat_id')
        outputs = self._model(**batch)
        labels[:, 0] = -100
        outputs = outputs[..., :self._tokenizer.vocab_size].contiguous()
        loss = self._calculator_loss(outputs, labels)
        return loss
    
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        strat_id = batch.pop('strat_id')
        outputs = self._model(**batch)
        loss = self._calculator_loss(outputs, labels)

        label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
        masked_lm_loss = torch.sum(loss) / torch.sum(label_size)
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=outputs,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def _calculator_loss(self, predict, labels):
        loss = F.cross_entropy(predict.view(-1, predict.size(-1)), labels.view(-1), reduction='none')
        loss = loss.view(labels.size(0), labels.size(1))

        return loss
    
    def _calculator_ppl_value(self, loss, labels):
        label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
        ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))
        return ppl_value

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
