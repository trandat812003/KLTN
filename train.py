import os
from lightning import Trainer
from dotenv import load_dotenv

from libs.dataset import MyDataModule
from modules.mymodule import MyModule


load_dotenv('./config/.env')


datamodule = MyDataModule(data_dir=os.getenv('DATA_DIR'))

model = MyModule()


trainer = Trainer(max_epochs=int(os.getenv("NUM_EPOCHS", 3)))
trainer.fit(model, datamodule=datamodule)

