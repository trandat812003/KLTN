import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv('./config/.env')


class Config:
    _DATA_NAME = os.getenv("DATA_NAME")
    _KNOWLEDGE_NAME = os.getenv("KNOWLEDGE_NAME")
    _BASELINE = os.getenv("BASELINE")
    _DATA_DIR = os.getenv("DATA_DIR")
    _NUM_EPOCHS = int(os.getenv("NUM_EPOCHS"))
    _BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    _GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS"))
    _MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH"))
    _MAX_DECODER_INPUT_LENGTH = int(os.getenv("MAX_DECODER_INPUT_LENGTH"))
    _WANDB_NAME = None

    @classmethod
    @property
    def DATA_NAME(cls):
        if cls._DATA_NAME is None:
            cls._DATA_NAME = 'esconv'
        return cls._DATA_NAME
    
    @classmethod
    @property
    def KNOWLEDGE_NAME(cls):
        if cls._KNOWLEDGE_NAME is None:
            cls._KNOWLEDGE_NAME = 'sbert'
        return cls._KNOWLEDGE_NAME
    
    @classmethod
    @property
    def BASELINE(cls):
        if cls._BASELINE is None:
            cls._BASELINE = 'kemi'
        return cls._BASELINE
    
    @classmethod
    @property
    def DATA_DIR(cls):
        if cls._DATA_DIR is None:
            cls._DATA_DIR = 'dataset/'
        return cls._DATA_DIR
    
    @classmethod
    @property
    def BATCH_SIZE(cls):
        if cls._BATCH_SIZE is None:
            cls._BATCH_SIZE = 25
        return cls._BATCH_SIZE
    
    @classmethod
    @property
    def GRADIENT_ACCUMULATION_STEPS(cls):
        if cls._GRADIENT_ACCUMULATION_STEPS is None:
            cls._GRADIENT_ACCUMULATION_STEPS = 1.0
        return cls._GRADIENT_ACCUMULATION_STEPS

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
