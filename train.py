import torch
from lightning import Trainer

from module import MyModule, MyDataModule
from libs.config import Config
from libs.utils.model_loader import get_tokenizer, get_model
from libs.config import Config
from libs.logging import Logging

torch.set_float32_matmul_precision('medium')
logging = Logging(stage="train", is_wandb=False)

tokenizer = get_tokenizer()
datamodule = MyDataModule(tokenizer=tokenizer)
model = get_model()
module = MyModule(tokenizer, model)

trainer = Trainer(
    max_epochs=Config.NUM_EPOCHS, 
    gradient_clip_val=Config.GRADIENT_ACCUMULATION_STEPS,
    accelerator='cpu',
)

trainer.fit(module, datamodule=datamodule)
trainer.test(module, datamodule=datamodule)
