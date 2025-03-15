import os
import yaml
from datetime import datetime

from libs.labels import labels_dataset_mapping


class Config:
    _WANDB_NAME = None
    DATA_NAME = "esconv"
    KNOWLEDGE_NAME = "sbert"
    BASELINE = "kemi"
    BATCH_SIZE = 5
    NUM_EPOCHS = 5
    GRADIENT_ACCUMULATION_STEPS = 1.0
    lr = 3e-5
    lr_agument = 1e-6

    @classmethod
    @property
    def WANDB_NAME(cls):
        if cls._WANDB_NAME is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cls._WANDB_NAME = f"{cls.KNOWLEDGE_NAME}_{timestamp}"
        return cls._WANDB_NAME

    @classmethod
    @property
    def MAX_INPUT_LENGTH(cls):
        if cls.BASELINE in ["kemi"]:
            return 256
        return 128

    @classmethod
    @property
    def MAX_DECODER_INPUT_LENGTH(cls):
        return 40


class BlenderbotConfig:
    config_path = "./config/blenderbot.yaml"

    if not os.path.exists(config_path):
        raise ValueError("Config file does not exist.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    PRETRAIN_MODEL = config["pretrained_model_path"]
    EXPANDED_VOCAB_DATA = config["expanded_vocab_data"]
    EXPANDED_VOCAB_KNOWLEDGE = config["expanded_vocab_knowledge"]
    CUSTOM_CONFIG_PATH = config["custom_config_path"]
    GRADIENT_CHECKPOINTING = config["gradient_checkpointing"]

    labels_mapping = labels_dataset_mapping[Config.DATA_NAME]

    @classmethod
    def select_strategy(cls, v: float, a: float, d: float) -> list[str]:
        from libs.utils.VAD_analyzer import VADAnalyzer

        return cls.labels_mapping[
            (
                VADAnalyzer.categorize_value(v),
                VADAnalyzer.categorize_value(a),
                VADAnalyzer.categorize_value(d),
            ),
            None,
        ]
