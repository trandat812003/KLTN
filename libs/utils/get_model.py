from transformers import AutoConfig
import torch
from modules.my_model import MyModel
from libs.config import BlenderbotConfig


def get_model(checkpoint: str=None, device=torch.device('cpu')):
    model = MyModel.from_pretrained(BlenderbotConfig.PRETRAIN_MODEL)
    if BlenderbotConfig.CUSTOM_CONFIG_PATH is not None:
        model = MyModel(AutoConfig.from_pretrained(BlenderbotConfig.CUSTOM_CONFIG_PATH))
    
    setattr(model.config, 'gradient_checkpointing', BlenderbotConfig.GRADIENT_CHECKPOINTING)
    
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    
    return model