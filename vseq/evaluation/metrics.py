import math

from typing import Optional, Set, Union

import torch


class Metric:
    base_tags = set()

    def __init__(
        self,
        name: str, tags:
        set):
        self.name = name
        self.tags = self.base_tags if tags is None else (tags | self.base_tags)

    @property
    def value(self):
        raise NotImplementedError()

    @property
    def str_value(self):
        raise NotImplementedError()

    def update(self, value: torch.Tensor):
        raise NotImplementedError()


class RunningMeanMetric(Metric):
    _str_value_fmt = "<10.3"

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str,
        tags: Set[str],
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        device: torch.device = None,
    ):
        super().__init__(name=name, tags=tags)

        values = torch.FloatTensor([values]) if isinstance(values, float) else values

        if isinstance(reduce_by, torch.Tensor):
            reduce_by = reduce_by.sum()
        elif isinstance(reduce_by, float):
            reduce_by = torch.FloatTensor([reduce_by])
        else:
            reduce_by = torch.FloatTensor([values.numel()])

        if isinstance(weight_by, torch.Tensor):
            weight_by = weight_by.sum()
        elif isinstance(weight_by, float):
            weight_by = torch.FloatTensor([weight_by])
        else:
            weight_by = reduce_by

        device = device or values.device

        values = values.to(device)
        reduce_by = reduce_by.to(device)
        weight_by = weight_by.to(device)

        values = values.sum() if isinstance(values, torch.Tensor) else values

        self.device = device
        self.weight_by = weight_by
        self._value = values / reduce_by  # Reduce within batch

    def to(self, device: torch.device):
        self.device = device
        self._value = self._value.to(device)
        self.weight_by = self.weight_by.to(device)
        return self

    @property
    def value(self):
        return self._value.item()

    @property
    def str_value(self):
        return f"{self.value:{self._str_value_fmt}f}"

    def update(self, metric: Metric):
        """Update the running mean statistic.

        Args:
            metric (RunningMeanMetric): The running mean metric to update with
        """
        d = self.weight_by + metric.weight_by
        w1 = self.weight_by / d
        w2 = metric.weight_by / d

        self._value = self._value * w1 + metric._value * w2  # Reduce between batches (over entire epoch)

        self.weight_by = d


class LLMetric(RunningMeanMetric):
    base_tags = {"log_likelihoods"}

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "ll",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        device: torch.device = None
    ):
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by, device=device)


class KLMetric(RunningMeanMetric):
    base_tags = {"kl_divergences"}

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "kl",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        device: torch.device = None
    ):
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by, device=device)


class BitsPerDimMetric(RunningMeanMetric):
    base_tags = set()

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "bpd",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        device: torch.device = None
    ):
        values = - values / math.log(2)
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by, device=device)


class PerplexityMetric(BitsPerDimMetric):
    """Perplexity computed as $2^{-\frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i)}$

    Args:
        RunningMeanMetric ([type]): [description]
    """

    base_tags = set()

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "pp",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        device: torch.device = None
    ):
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by, device=device)

    @property
    def value(self):
        return 2 ** self._value
