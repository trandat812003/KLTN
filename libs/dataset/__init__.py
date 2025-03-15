from typing import Type

from libs.dataset.base import BaseDataset
from libs.dataset.esconv import ESConvDataset
from libs.dataset.mi import MIDataset
from libs.dataset.augment import AugmentDataset
from libs.config import Config


MyDataset: Type[BaseDataset]
if Config.DATA_NAME == "esconv":
    MyDataset = ESConvDataset
else:
    MyDataset = MIDataset


__all__ = ["MyDataset", "BaseDataset", "AugmentDataset"]
