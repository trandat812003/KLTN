import wandb
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

    def log(self, d: dict):
        wandb.log(d)