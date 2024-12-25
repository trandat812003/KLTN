import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv('./config/.env')


class Config:
    DATA_NAME = os.getenv("DATA_NAME")
    KNOWLEDGE_NAME = os.getenv("KNOWLEDGE_NAME")
    BASELINE = os.getenv("BASELINE")
    DATA_DIR = os.getenv("DATA_DIR")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS"))
    _WANDB_NAME = None

    @classmethod
    @property
    def WANDB_NAME(cls):
        if cls._WANDB_NAME is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cls._WANDB_NAME = f'{cls.KNOWLEDGE_NAME}_{timestamp}'
        return cls._WANDB_NAME
    
    @classmethod
    @property
    def NUM_EPOCHS(cls):
        if 'pal' in cls.BASELINE:
            return 10
        return 5
    
    @classmethod
    @property
    def lr(cls):
        if 'pal' in cls.BASELINE:
            return 1.5e-5
        return 3e-5

    @classmethod
    @property
    def MAX_INPUT_LENGTH(cls):
        if 'pal' in cls.BASELINE:
            return 512
        return 256

    @classmethod
    @property
    def MAX_DECODER_INPUT_LENGTH(cls):
        if 'pal' in cls.BASELINE:
            return 50
        return 40
