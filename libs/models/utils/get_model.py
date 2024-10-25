import yaml
from transformers import AutoTokenizer, AutoConfig
import torch
from libs.models.blenderbot import ModelBlenderbot


def get_model(data_name: str, knowledge_name: str, tokenizer: AutoTokenizer, checkpoint: str):
    with open('./config/blenderbot.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model = ModelBlenderbot.from_pretrained(config['pretrained_model_path'])
    if config.get('custom_config_path', None) is not None:
        model = ModelBlenderbot(AutoConfig.from_pretrained(config['custom_config_path']))
    
    if 'gradient_checkpointing' in config:
        setattr(model.config, 'gradient_checkpointing', config['gradient_checkpointing'])
    
    if 'expanded_vocab' in config:
        expanded_vocab = config['expanded_vocab'][data_name]
        if knowledge_name != 'none':
            expanded_vocab += config['expanded_vocab'][knowledge_name]
        tokenizer.add_tokens(expanded_vocab, special_tokens=True)
    model.tie_tokenizer(tokenizer)
    
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))
    
    return model, tokenizer