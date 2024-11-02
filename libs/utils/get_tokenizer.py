from transformers import AutoTokenizer
from libs.config import Config, BlenderbotConfig


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BlenderbotConfig.PRETRAIN_MODEL)

    expanded_vocab = BlenderbotConfig.EXPANDED_VOCAB_DATA[Config.DATA_NAME]
    expanded_vocab += BlenderbotConfig.EXPANDED_VOCAB_KNOWLEDGE[Config.KNOWLEDGE_NAME]

    tokenizer.add_tokens(expanded_vocab, special_tokens=True)

    return tokenizer
