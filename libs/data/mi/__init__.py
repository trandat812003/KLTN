from typing import Any
from typing_extensions import Dict


class MIDataset:
    def __init__(self, data: Dict) -> None:
        self._data = data

    def __getattribute__(self, name: str) -> str:
        return self._data[name]