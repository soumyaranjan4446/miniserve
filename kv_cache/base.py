from __future__ import annotations
from abc import ABC, abstractmethod

import torch


class KVCache(ABC):
    @abstractmethod
    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor): ...
    @abstractmethod
    def get(self, layer_idx: int): ...
    @abstractmethod
    def get_seq_length(self) -> int: ...
    @abstractmethod
    def reset(self): ...
