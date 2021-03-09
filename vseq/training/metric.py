from typing import List, Set, Union
from dataclasses import dataclass

import torch


@dataclass
class Metric:
    name: str
    value: Union[torch.Tensor, float, int]
    tags: Set[str] = None
    aggregate_method: str = 'concatenate'

    def set_tags(self, tags: Set[str]):
        self.tags = tags

    def update_tags(self, tags: Set[str]):
        self.tags |= tags
