import os
import yaml
from transformers import AutoTokenizer


def get_tokenizer(data_name: str, knowledge_name: str):
    with open('./config/data.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if 'model_name' not in config or 'pretrained_model_path' not in config:
        raise ValueError("Invalid configuration format: 'model_name' or 'pretrained_model_path' missing.")

    toker = AutoTokenizer.from_pretrained(config['pretrained_model_path'])

    if 'expanded_vocab' in config:
        expanded_vocab = config['expanded_vocab'][data_name]
        if knowledge_name != 'none':
            expanded_vocab += config['expanded_vocab'][knowledge_name]
        toker.add_tokens(expanded_vocab, special_tokens=True)
    return toker


def get_model():
    pass