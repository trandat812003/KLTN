from lightning import Trainer
from modules.my_module import MyModule
from modules.my_datamodule import MyDataModule
from libs.config import Config
from libs.utils.get_tokenizer import get_tokenizer


datamodule = MyDataModule(tokenizer=get_tokenizer())

model = MyModule()


trainer = Trainer(max_epochs=Config.BATCH_SIZE)
trainer.fit(model, datamodule=datamodule)