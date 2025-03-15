import torch
from src.transformers import AutoTokenizer, AutoConfig

from libs.config import BlenderbotConfig, Config
from module import MyModel


def get_model():
    model = MyModel.from_pretrained(BlenderbotConfig.PRETRAIN_MODEL)

    if BlenderbotConfig.CUSTOM_CONFIG_PATH is not None:
        model = MyModel(AutoConfig.from_pretrained(BlenderbotConfig.CUSTOM_CONFIG_PATH))

    setattr(
        model.config, "gradient_checkpointing", BlenderbotConfig.GRADIENT_CHECKPOINTING
    )

    return model


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BlenderbotConfig.PRETRAIN_MODEL)

    expanded_vocab = BlenderbotConfig.EXPANDED_VOCAB_DATA[Config.DATA_NAME]
    expanded_vocab += BlenderbotConfig.EXPANDED_VOCAB_KNOWLEDGE[Config.KNOWLEDGE_NAME]

    tokenizer.add_tokens(expanded_vocab, special_tokens=True)

    return tokenizer
