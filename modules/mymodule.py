import lightning as L
import os
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from dotenv import load_dotenv

from libs.utils import get_model, get_tokenizer


load_dotenv('./config/.env')

class MyModule(L.LightningModule):
    def __init__(self):
        super().__init__()

        data_name = os.getenv("DATA_NAME")
        knowledge_name = os.getenv('KNOWLEDGE_NAME')
        checkpoint = os.getenv('CHECKPOINT')

        self._tokenizer = get_tokenizer(data_name=data_name, knowledge_name=knowledge_name)
        self._model, self._tokenizer = get_model(
            data_name=data_name, 
            knowledge_name=knowledge_name, 
            tokenizer=self._tokenizer,
        )

        self.save_hyperparameters()

        self._model.to(self.device)

    def training_step(self, batch, batch_idx):
        breakpoint()
        outputs = self._model(**batch)

        breakpoint()
        loss = outputs.pop('all')
        ppl = outputs.pop('ppl')



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