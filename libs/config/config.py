import os
from dotenv import load_dotenv

load_dotenv('./config/.env')


class Config:
    DATA_NAME = os.getenv("DATA_NAME")
    KNOWLEDGE_NAME = os.getenv("KNOWLEDGE_NAME")
    BASELINE = os.getenv("BASELINE")
    DATA_DIR = os.getenv("DATA_DIR")
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS"))
    MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH"))
    MAX_DECODER_INPUT_LENGTH = int(os.getenv("MAX_DECODER_INPUT_LENGTH"))
