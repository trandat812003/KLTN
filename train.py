from lightning import Trainer
from modules.my_module import MyModule
from modules.my_datamodule import MyDataModule
from libs.config import Config
from libs.utils.get_tokenizer import get_tokenizer
from libs.utils.get_model import get_model

tokenizer = get_tokenizer()
datamodule = MyDataModule(tokenizer=tokenizer)
model = get_model("cpu")
module = MyModule(tokenizer, model)

trainer = Trainer(
    max_epochs=Config.NUM_EPOCHS, 
    gradient_clip_val=Config.GRADIENT_ACCUMULATION_STEPS,
    accelerator="cpu",
)
trainer.fit(module, datamodule=datamodule)
trainer.test(module, datamodule=datamodule)
