import math

from typing import Optional, Union, List

import torch


class Metric:
    base_tags = set()

    def __init__(self, name: str, tags: set, accumulate: bool, keep_on_device: bool):
        self.name = name
        self.tags = self.base_tags if tags is None else (tags | self.base_tags)
        self.accumulate = accumulate
        self.keep_on_device = keep_on_device
        self.accumulated_values = []

    @property
    def log_value(self):
        raise NotImplementedError()

    @property
    def str_value(self):
        raise NotImplementedError()

    def _accumulate(self, value: Union[torch.Tensor, float]):
        if not self.keep_on_device:
            value = value.cpu()

        if isinstance(value, torch.Tensor):
            value = value.tolist()

        self.accumulated_values.append(value)

    def update(self, value: torch.Tensor):
        raise NotImplementedError()

    def reset(self):
        self.accumulated_values = []


class RunningMean(Metric):
    _str_value_fmt = "<10.3"

    def __init__(self, name: str, tags: set, accumulate: bool = False, keep_on_device: bool = False):
        """RunningMean computes a cumulative sum by default and weighted sum via the `weight` argument.

        If a batch of metrics are passed to `update` it first reduces by the mean.

        Args:
            name (str): Name of the metric
            tags (set): Set of tags of the metric
            accumulate (bool, optional): If True, keep all values passed to `update`. Defaults to False.
            keep_on_device (bool, optional): If True, keep all values on their device. Defaults to False.
        """
        super().__init__(name=name, tags=tags, accumulate=accumulate, keep_on_device=keep_on_device)
        self._running = 0
        self._weight = 0

    @property
    def log_value(self):
        return self._running

    @property
    def str_value(self):
        return f"{self.log_value:{self._str_value_fmt}f}"

    def update(
        self,
        value: torch.Tensor,
        reduce_by: Optional[torch.Tensor] = None,
        weight_by: Optional[torch.Tensor] = None,
    ):
        """Update the running statistic (running mean by default).

        Args:
            value (torch.Tensor): Value by which to update the statistic.
            reduce_by (Optional[torch.Tensor]): Weight used to reduce `value`. Defaults to `value.numel()`.
            weight_by (Optional[torch.Tensor]): Weight of reduced `value` in the running statistic. Defaults to `reduce_by`.
        """

        if self.accumulate:
            self.accumulate(value)
        
        reduce_by = value.numel() if reduce_by is None else reduce_by.sum().cpu().item()
        weight_by = weight_by or reduce_by

        value = value.sum().cpu().item() / reduce_by  # Reduce within batch

        self._weight += weight_by

        weight_value = weight_by / self._weight
        weight_running = (self._weight - weight_by) / self._weight
        self._running = (
            value * weight_value + self._running * weight_running
        )  # Reduce between batches (over entire epoch)

    def reset(self):
        super().reset()
        self._running = 0
        self._weight = 0


class LLMetric(RunningMean):
    base_tags = {"likelihoods"}

    def __init__(self, name: str = "ll", tags: set = None, accumulate: bool = False, keep_on_device: bool = False):
        super().__init__(name=name, tags=tags, accumulate=accumulate, keep_on_device=keep_on_device)


class KLMetric(RunningMean):
    base_tags = {"kl_divergences"}

    def __init__(self, name: str = "kl", tags: set = None, accumulate: bool = False, keep_on_device: bool = False):
        super().__init__(name=name, tags=tags, accumulate=accumulate, keep_on_device=keep_on_device)


class BitsPerDimMetric(RunningMean):
    base_tags = set()
    # _str_value_fmt = "<5.2"

    def __init__(self, name: str = "bpd", tags: set = None, accumulate: bool = False, keep_on_device: bool = False):
        super().__init__(name, tags, accumulate=accumulate, keep_on_device=keep_on_device)

    def update(
        self,
        log_likelihood: Union[torch.Tensor, float],
        reduce_by: Optional[torch.Tensor] = None,
        weight_by: Optional[torch.Tensor] = None,
    ):

        log2_likelihood = -log_likelihood / math.log(2)
        super().update(log2_likelihood, reduce_by=reduce_by, weight_by=weight_by)


class PerplexityMetric(BitsPerDimMetric):
    """Perplexity computed as $2^{-\frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i)}$

    Args:
        RunningMean ([type]): [description]
    """

    base_tags = set()
    # _str_value_fmt = "<8.2"

    def __init__(self, name: str = "pp", tags: set = None, accumulate: bool = False, keep_on_device: bool = False):
        super().__init__(name=name, tags=tags, accumulate=accumulate, keep_on_device=keep_on_device)

    @property
    def log_value(self):
        return 2 ** self._running
