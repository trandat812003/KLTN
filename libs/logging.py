import os
import wandb
import csv
import logging
from libs.utils import create_folder
from libs.config import Config

class Logging:
    def __init__(self, is_wandb: bool = True, is_predict: bool = False):
        self.is_wandb = is_wandb
        self.is_predict = is_predict
        if is_wandb:
            wandb.init(
                project=f'{Config.BASELINE}', 
                name=Config.WANDB_NAME,
                config={
                    "learning_rate": Config.lr,
                    "epochs": Config.NUM_EPOCHS,
                },
            )
            #58bfd6403a8f96bd35e09d284fc38f5aae23604f

        create_folder(f'./logs/{Config.BASELINE}')
        log_filename = f'./logs/{Config.BASELINE}/{Config.WANDB_NAME}.log'
        logging.basicConfig(
            filename=log_filename,
            filemode='a',  # 'a' để ghi tiếp vào tệp, 'w' để ghi đè
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger()

        create_folder(f'./logs_csv/{Config.BASELINE}')

        if is_predict:
            self.csv_path = f'./logs_csv/{Config.BASELINE}/{Config.WANDB_NAME}_predict.csv'
        else:
            self.csv_path = f'./logs_csv/{Config.BASELINE}/{Config.WANDB_NAME}.csv'
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["epoch", "phase", "loss", "ppl"])

    def log(self, d: dict):
        if self.is_wandb:
            wandb.log(d)
        self.logger.info(d)

    def log_csv(self, **kwargs):
        with open(self.csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            if self.is_predict:
                writer.writerow([f"ref: {kwargs["ref"]}"])
                writer.writerow([f"gen: {kwargs["gen"]}"])
                writer.writerow([])
            else:
               writer.writerow([kwargs["epoch"], kwargs["phase"], kwargs["ppl"]]) 


logging = Logging()