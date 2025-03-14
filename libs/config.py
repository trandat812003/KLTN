import os
import yaml
from datetime import datetime


class BlenderbotConfig:
    config_path = './config/blenderbot.yaml'

    if not os.path.exists(config_path):
        raise ValueError("Config file does not exist.")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    PRETRAIN_MODEL = config['pretrained_model_path']
    EXPANDED_VOCAB_DATA = config['expanded_vocab_data']
    EXPANDED_VOCAB_KNOWLEDGE = config['expanded_vocab_knowledge']
    CUSTOM_CONFIG_PATH = config['custom_config_path']
    GRADIENT_CHECKPOINTING = config['gradient_checkpointing']

    _QUESTION = "[Question]"
    _RESTATEMENT = "[Restatement or Paraphrasing]"
    _REFLECTION = "[Reflection of feelings]"
    _SELF_DISCLOSURE = "[Self-disclosure]"
    _AFFIRMATION = "[Affirmation and Reassurance]"
    _SUGGESTION = "[Providing Suggestions]"
    _INFORMATION = "[Information]"
    _OTHERS = "[Others]"

    _GIV = "[GIV]"
    _QUEST = "[QUEST]"
    _SEEK = "[SEEK]"
    _AF = "[AF]"
    _PWP = "[PWP]"
    _PWOP = "[PWOP]"
    _EMPH = "[EMPH]"
    _CON = "[CON]"
    _SR = "[SR]"
    _CR = "[CR]"

    @classmethod
    def select_strategy(cls, v: float, a: float, d: float) -> list[str]:
        if Config.DATA_NAME in ["esconv"]:
            if v > 0.6 and d > 0.5:
                return [cls._AFFIRMATION, cls._SUGGESTION, cls._INFORMATION]

            if v < 0.3 and a > 0.6 and d < 0.4:
                return [cls._REFLECTION]

            if a > 0.7 and d < 0.4:
                return [cls._SELF_DISCLOSURE]

            if d > 0.6 and a < 0.5:
                return [cls._RESTATEMENT]

            if 0.5 < a and 0.5 < d and v < 0.6:
                return [cls._QUESTION]

            return [cls._OTHERS]
        else:
            if v > 0.6 and d > 0.5:
                return [cls._GIV]

            if 0.5 < a and 0.5 < d and v < 0.6:
                return [cls._QUEST]

            if a > 0.7 and d < 0.4:
                return [cls._AF]

            if d > 0.6 and a < 0.5:
                return [cls._PWP]

            if d > 0.6 and a > 0.5:
                return [cls._PWOP]

            if v < 0.3 and a > 0.6 and d < 0.4:
                return [cls._EMPH]

            if a > 0.6 and d > 0.6:
                return [cls._CON]

            if d > 0.6 and a < 0.5:
                return [cls._SR]

            if v < 0.5 and d > 0.5:
                return [cls._CR]
            
            return [cls._SEEK]


class Config:
    _WANDB_NAME = None
    DATA_NAME = "esconv"
    KNOWLEDGE_NAME = "sbert"
    BASELINE = "kemi"
    BATCH_SIZE = 5
    NUM_EPOCHS =5
    GRADIENT_ACCUMULATION_STEPS = 1.0
    lr = 3e-5
    lr_agument = 1e-6


    @classmethod
    @property
    def WANDB_NAME(cls):
        if cls._WANDB_NAME is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cls._WANDB_NAME = f'{cls.KNOWLEDGE_NAME}_{timestamp}'
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
