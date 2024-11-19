
import torch
import lightning as L
import torch.nn.functional as F
import numpy as np
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import f1_score

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

        self.train_loss = 0.0
        self.train_ppl = 0.0
        self.train_size = 0
        self.train_steps = 0

        self.test_loss = 0.0
        self.test_ppl = 0.0
        self.test_size = 0
        self.test_steps = 0

        self.val_loss = 0.0
        self.val_ppl = 0.0
        self.val_size = 0
        self.val_steps = 0
        self.val_metrics = {'bleu1': [], 'bleu2': [], 'bleu4': [], 'wf1': []}
        self.test_metrics = {'bleu1': [], 'bleu2': [], 'bleu4': [], 'wf1': []}

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        strat_id = batch.pop('strat_id')
        outputs = self._model(**batch)
        loss, ppl = self._calculator_loss_and_ppl_value(outputs, labels)

        input_ids = batch['input_ids']

        tmp_loss = float(loss.item()) * (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS / input_ids.shape[0])
        self.train_loss += tmp_loss
        self.train_size += input_ids.size(0)
        self.train_steps += 1

        tmp_ppl = ppl.item() if ppl.item() < float('inf') else self.train_ppl
        self.train_ppl += tmp_ppl

        return loss

    def on_train_epoch_end(self):
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self.log("train_loss", self.train_loss / self.train_steps, prog_bar=True)
        self.log("train_ppl", np.exp(self.train_loss / self.train_steps), prog_bar=True)
        self.train_loss = 0.0
        self.train_ppl = 0.0
        self.train_size = 0
        self.train_steps = 0

    
    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        strat_id = batch.pop('strat_id')
        outputs = self._model(**batch)
        labels[:, 0] = -100
        outputs = outputs[..., :self._tokenizer.vocab_size].contiguous()
        loss, ppl = self._calculator_loss_and_ppl_value(outputs, labels)

        input_ids = batch['input_ids']

        tmp_loss = float(loss.item()) * (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS / input_ids.shape[0])
        self.val_loss += tmp_loss
        self.val_size += input_ids.size(0)
        self.val_steps += 1

        tmp_ppl = ppl.item() if ppl.item() < float('inf') else self.val_ppl
        self.val_ppl += tmp_ppl

        predicted_ids = torch.argmax(outputs, dim=-1).cpu().tolist()
        label_ids = labels.cpu().tolist()

        # Tính BLEU và WF1
        bleu1 = self._calculate_bleu(label_ids, predicted_ids, n_gram=1)
        bleu2 = self._calculate_bleu(label_ids, predicted_ids, n_gram=2)
        bleu4 = self._calculate_bleu(label_ids, predicted_ids, n_gram=4)
        wf1 = self._calculate_wf1(label_ids, predicted_ids)

        # Lưu giá trị vào biến val_metrics
        self.val_metrics['bleu1'].append(bleu1)
        self.val_metrics['bleu2'].append(bleu2)
        self.val_metrics['bleu4'].append(bleu4)
        self.val_metrics['wf1'].append(wf1)

        return loss
    
    def on_validation_epoch_end(self):
        self.log("validate_loss", self.val_loss / self.val_steps, prog_bar=True)
        self.log("validate_ppl", np.exp(self.val_loss / self.val_steps), prog_bar=True)

        avg_bleu1 = sum(self.val_metrics['bleu1']) / len(self.val_metrics['bleu1'])
        avg_bleu2 = sum(self.val_metrics['bleu2']) / len(self.val_metrics['bleu2'])
        avg_bleu4 = sum(self.val_metrics['bleu4']) / len(self.val_metrics['bleu4'])
        avg_wf1 = sum(self.val_metrics['wf1']) / len(self.val_metrics['wf1'])

        # Log giá trị
        self.log("epoch_bleu_1", avg_bleu1, prog_bar=True)
        self.log("epoch_bleu_2", avg_bleu2, prog_bar=True)
        self.log("epoch_bleu_4", avg_bleu4, prog_bar=True)
        self.log("epoch_wf1", avg_wf1, prog_bar=True)

        # Reset metrics
        self.val_metrics = {'bleu1': [], 'bleu2': [], 'bleu4': [], 'wf1': []}

        self.val_loss = 0.0
        self.val_ppl = 0.0
        self.val_size = 0
        self.val_steps = 0

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        strat_id = batch.pop('strat_id')
        outputs = self._model(**batch)

        loss, ppl = self._calculator_loss_and_ppl_value(outputs, labels)

        input_ids = batch['input_ids']

        tmp_loss = float(loss.item()) * (Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS / input_ids.shape[0])
        self.test_loss += tmp_loss
        self.test_size += input_ids.size(0)
        self.test_steps += 1

        tmp_ppl = ppl.item() if ppl.item() < float('inf') else self.test_ppl
        self.test_ppl += tmp_ppl

        predicted_ids = torch.argmax(outputs, dim=-1).cpu().tolist()
        label_ids = labels.cpu().tolist()

        # Tính BLEU và WF1
        bleu1 = self._calculate_bleu(label_ids, predicted_ids, n_gram=1)
        bleu2 = self._calculate_bleu(label_ids, predicted_ids, n_gram=2)
        bleu4 = self._calculate_bleu(label_ids, predicted_ids, n_gram=4)
        wf1 = self._calculate_wf1(label_ids, predicted_ids)

        # Lưu giá trị vào biến test_metrics
        self.test_metrics['bleu1'].append(bleu1)
        self.test_metrics['bleu2'].append(bleu2)
        self.test_metrics['bleu4'].append(bleu4)
        self.test_metrics['wf1'].append(wf1)

        return loss

    
    def on_test_epoch_end(self):
        self.log("test_loss", self.test_loss / self.test_size, prog_bar=True)
        self.log("test_ppl", np.exp(self.test_loss / self.test_size), prog_bar=True)

        # Tính trung bình BLEU và WF1
        avg_bleu1 = sum(self.test_metrics['bleu1']) / len(self.test_metrics['bleu1'])
        avg_bleu2 = sum(self.test_metrics['bleu2']) / len(self.test_metrics['bleu2'])
        avg_bleu4 = sum(self.test_metrics['bleu4']) / len(self.test_metrics['bleu4'])
        avg_wf1 = sum(self.test_metrics['wf1']) / len(self.test_metrics['wf1'])

        # Log giá trị
        self.log("test_bleu_1", avg_bleu1, prog_bar=True)
        self.log("test_bleu_2", avg_bleu2, prog_bar=True)
        self.log("test_bleu_4", avg_bleu4, prog_bar=True)
        self.log("test_wf1", avg_wf1, prog_bar=True)

        # Reset metrics
        self.test_metrics = {'bleu1': [], 'bleu2': [], 'bleu4': [], 'wf1': []}
        
        self.test_loss = 0.0
        self.test_ppl = 0.0
        self.test_size = 0
        self.test_size = 0

    def _calculator_loss_and_ppl_value(self, predict, labels):
        loss = F.cross_entropy(predict.view(-1, predict.size(-1)), labels.view(-1), reduction='none')
        loss = loss.view(labels.size(0), labels.size(1))
        label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
        ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))
        loss = torch.sum(loss) / torch.sum(label_size)

        return loss, ppl_value
    
    def _calculate_bleu(self, references, predictions, n_gram=4):
        weights = [1.0 / n_gram] * n_gram  # Tính điểm BLEU với n_gram
        bleu_scores = []
        chencherry = SmoothingFunction()

        for ref, pred in zip(references, predictions):
            score = sentence_bleu(
                [ref], pred, weights=weights, smoothing_function=chencherry.method1
            )
            bleu_scores.append(score)

        return sum(bleu_scores) / len(bleu_scores)
    
    def _calculate_wf1(self, references, predictions):
        references = [item for sublist in references for item in sublist]  # Flatten danh sách
        predictions = [item for sublist in predictions for item in sublist]  # Flatten danh sách

        return f1_score(references, predictions, average='weighted')

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
