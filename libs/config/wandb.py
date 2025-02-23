import wandb
import os
import csv
from libs.config import Config
from libs.utils.file_manager import create_folder


class Logging:
    def __init__(self):
        wandb.init(
            project=f'{Config.BASELINE}', 
            name=Config.WANDB_NAME,
            config={
                "learning_rate": 3e-5,
                "epochs": Config.NUM_EPOCHS,
            },
        )
        #58bfd6403a8f96bd35e09d284fc38f5aae23604f

        self.csv_path = f'./logs_csv/{Config.BASELINE}/{Config.WANDB_NAME}.csv'
        self.predict_path = f'./logs_csv/{Config.BASELINE}/{Config.WANDB_NAME}_predict.csv'
        create_folder(f'./logs_csv/{Config.BASELINE}')
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "phase", "loss", "ppl"])

        

    def log(self, d: dict):
        wandb.log(d)

    def log_csv(self, epoch, phase, loss, ppl):
        with open(self.csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, phase, loss, ppl])

    def log_predict(self, ref: str, gen: str) -> None:
        with open(self.predict_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([f"ref: {ref}"])
            writer.writerow([f"gen: {gen}"])
            writer.writerow([])
