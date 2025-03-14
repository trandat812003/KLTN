import os
import wandb
import csv
from libs.utils import create_folder
from libs.config import Config

class Logging:
    def __init__(self, stage: str = None, is_wandb: bool = True):
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

        create_folder(f'./logs_csv/{Config.BASELINE}')

        if stage == "predict":
            self.csv_path = f'./logs_csv/{Config.BASELINE}/{Config.WANDB_NAME}_predict.csv'
        else:
            self.csv_path = f'./logs_csv/{Config.BASELINE}/{Config.WANDB_NAME}.csv'
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["epoch", "phase", "loss", "ppl"])

    def log(self, d: dict):
        wandb.log(d)

    def log_csv(self, **kwargs):
        with open(self.csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            if kwargs["stage"] == "predict":
                writer.writerow([f"ref: {kwargs["ref"]}"])
                writer.writerow([f"gen: {kwargs["gen"]}"])
                writer.writerow([])
            else:
               writer.writerow([kwargs["epoch"], kwargs["phase"], kwargs["ppl"]]) 
