import wandb
import os
import csv
from libs.config import Config


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

        self.csv_path = f'./logs_csv/{Config.WANDB_NAME}.csv'
        if not os.path.exists('logs_csv'):
            os.makedirs('logs_csv')
        with open(self.csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "phase", "loss", "ppl"])

    def log(self, d: dict):
        wandb.log(d)

    def log_csv(self, epoch, phase, loss, ppl):
        with open(self.csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, phase, loss, ppl])