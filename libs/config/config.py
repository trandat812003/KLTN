import os
from dotenv import load_dotenv

load_dotenv('./config/.env')


class Config:
    DATA_NAME = os.getenv("DATA_NAME")
    KNOWLEDGE_NAME = os.getenv("KNOWLEDGE_NAME")
    BASELINE = os.getenv("BASELINE")
    CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
    DATA_DIR = os.getenv("DATA_DIR")
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS"))
