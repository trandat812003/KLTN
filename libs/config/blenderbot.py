import os
import yaml


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

