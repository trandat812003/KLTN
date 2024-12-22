import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv('./config/.env')


class Config:
    DATA_NAME = os.getenv("DATA_NAME")
    KNOWLEDGE_NAME = os.getenv("KNOWLEDGE_NAME")
    BASELINE = os.getenv("BASELINE")
    DATA_DIR = os.getenv("DATA_DIR")
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS"))
    # MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH"))
    # MAX_DECODER_INPUT_LENGTH = int(os.getenv("MAX_DECODER_INPUT_LENGTH"))
    _MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH"))
    _MAX_DECODER_INPUT_LENGTH = int(os.getenv("MAX_DECODER_INPUT_LENGTH"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    WANDB_NAME = f'{KNOWLEDGE_NAME}_{timestamp}'

    @classmethod
    @property
    def MAX_INPUT_LENGTH(cls):
        if cls.BASELINE in ['kemi']:
            return 256
        return 512

    @classmethod
    @property
    def MAX_DECODER_INPUT_LENGTH(cls):
        if cls.BASELINE in ['kemi']:
            return 40
        return 50
