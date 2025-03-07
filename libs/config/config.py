import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv('./config/.env')


class Config:
    _WANDB_NAME = None

    @classmethod
    @property
    def DATA_NAME(cls):
        return 'esconv'
    
    @classmethod
    @property
    def KNOWLEDGE_NAME(cls):
        return 'sbert'
    
    @classmethod
    @property
    def BASELINE(cls):
        return 'kemi'
    
    @classmethod
    @property
    def BATCH_SIZE(cls):
        return 5
    
    @classmethod
    @property
    def GRADIENT_ACCUMULATION_STEPS(cls):
        return 1.0

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
        return 5
    
    @classmethod
    @property
    def lr(cls):
        return 3e-5


    @classmethod
    @property
    def MAX_INPUT_LENGTH(cls):
        return 128

    @classmethod
    @property
    def MAX_DECODER_INPUT_LENGTH(cls):
        return 40
