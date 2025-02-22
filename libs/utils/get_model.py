from transformers import AutoConfig
import torch
import torch.nn as nn
from modules.my_model import MyModel
from libs.config import BlenderbotConfig


def get_model(device=torch.device('cpu'), checkpoint: str=None):
    model = MyModel.from_pretrained(BlenderbotConfig.PRETRAIN_MODEL)

    if BlenderbotConfig.CUSTOM_CONFIG_PATH is not None:
        model = MyModel(AutoConfig.from_pretrained(BlenderbotConfig.CUSTOM_CONFIG_PATH))
    
    setattr(model.config, 'gradient_checkpointing', BlenderbotConfig.GRADIENT_CHECKPOINTING)

    model.strategy_alpha = nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    
    return model