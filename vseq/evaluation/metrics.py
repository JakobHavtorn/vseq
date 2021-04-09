import math

from typing import Optional, Set, Union

import torch

from vseq.utils.operations import detach_to_device, infer_device


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
        weight_by: Optional[Union[torch.Tensor, float]] = None
    ):
        super().__init__(name=name, tags=tags)

        numel = values.numel() if isinstance(values, torch.Tensor) else 1
        value = values.sum().tolist() if isinstance(values, torch.Tensor) else values

        reduce_by = reduce_by.sum().tolist() if isinstance(reduce_by, torch.Tensor) else (reduce_by or numel)

        weight_by = weight_by.sum().tolist() if isinstance(weight_by, torch.Tensor) else (weight_by or reduce_by)

        self.weight_by = weight_by
        self._value = value / reduce_by

    @property
    def value(self):
        return self._value

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
        weight_by: Optional[Union[torch.Tensor, float]] = None
    ):
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by)


class KLMetric(RunningMeanMetric):
    base_tags = {"kl_divergences"}

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "kl",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None
    ):
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by)


class BitsPerDimMetric(RunningMeanMetric):
    base_tags = set()

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "bpd",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None
    ):
        values = - values / math.log(2)
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by)


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
        weight_by: Optional[Union[torch.Tensor, float]] = None
    ):
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by)

    @property
    def value(self):
        return 2 ** self._value
