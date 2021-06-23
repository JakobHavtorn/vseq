import math

from copy import deepcopy
from typing import List, Optional, Set, Union

import torch

from vseq.utils.operations import detach


class Metric:
    base_tags = set()
    _str_value_fmt = "<.3"

    def __init__(
        self,
        name: str,
        tags: Set[str] = None,
        get_best: str = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        self.name = name
        self.tags = self.base_tags if tags is None else (tags | self.base_tags)
        self.get_best = GET_BEST[get_best] if get_best is not None else GET_BEST["none"]
        self.log_to_console = log_to_console
        self.log_to_framework = log_to_framework

    @property
    def value(self):
        """Primary value of the metric to be used for logging"""
        raise NotImplementedError()

    @property
    def str_value(self):
        return f"{self.value:{self._str_value_fmt}f}"

    def update(self, metric):
        """Update the metric (e.g. running mean)"""
        raise NotImplementedError()

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, value={self.str_value})"


def min_value(metrics: List[Metric]):
    return min(metrics, key=lambda m: m.value)


def max_value(metrics: List[Metric]):
    return max(metrics, key=lambda m: m.value)


def no_value(metrics: List[Metric]):
    return None


GET_BEST = dict(none=no_value, min=min_value, max=max_value)


class LatestMeanMetric(Metric):
    _str_value_fmt = "<.3"

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str,
        tags: Set[str] = None,
        get_best: str = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
    ):
        """Create a latest mean metric that maintains the latest mean when updated.

        Args:
            values (Union[torch.Tensor, float]): Values of the metric
            name (str): Name of the metric
            tags (Set[str]): Tags to use for grouping with other metrics.
            reduce_by (Optional[Union[torch.Tensor, float]], optional): A single or per example divisor of the values. Defaults to batch size.
        """
        super().__init__(name=name, tags=tags, get_best=get_best)

        values = detach(values)
        reduce_by = detach(reduce_by)

        numel = values.numel() if isinstance(values, torch.Tensor) else 1
        value = values.sum().tolist() if isinstance(values, torch.Tensor) else values

        reduce_by = reduce_by.sum().tolist() if isinstance(reduce_by, torch.Tensor) else (reduce_by or numel)

        self._value = value / reduce_by

    @property
    def value(self):
        return self._value

    def update(self, metric: Metric):
        self._value = metric.value


class RunningMeanMetric(Metric):
    _str_value_fmt = "<.3"

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str,
        tags: Set[str] = None,
        get_best: str = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        """Create a running mean metric that maintains the running mean when updated.

        Args:
            values (Union[torch.Tensor, float]): Values of the metric
            name (str): Name of the metric
            tags (Set[str]): Tags to use for grouping with other metrics.
            reduce_by (Optional[Union[torch.Tensor, float]], optional): A single or per example divisor of the values. Defaults to batch size.
            weight_by (Optional[Union[torch.Tensor, float]], optional): A single or per example weights for the running mean. Defaults to `reduce_by`.
        """
        super().__init__(
            name=name, tags=tags, get_best=get_best, log_to_console=log_to_console, log_to_framework=log_to_framework
        )

        values = detach(values)
        reduce_by = detach(reduce_by)
        weight_by = detach(weight_by)

        numel = values.numel() if isinstance(values, torch.Tensor) else 1
        value = values.sum().tolist() if isinstance(values, torch.Tensor) else values

        reduce_by = reduce_by.sum().tolist() if isinstance(reduce_by, torch.Tensor) else (reduce_by or numel)

        weight_by = weight_by.sum().tolist() if isinstance(weight_by, torch.Tensor) else (weight_by or reduce_by)

        self.weight_by = weight_by
        self._value = value / reduce_by

    @property
    def value(self):
        return self._value

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


class AccuracyMetric(Metric):
    _str_value_fmt = "6.4"  # 6.4321

    def __init__(
        self,
        predictions: Union[torch.Tensor, float],
        labels: Union[torch.Tensor, float],
        name: str = "accuracy",
        tags: Set[str] = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        """Standard classification accuracy"""
        super().__init__(
            name=name, tags=tags, get_best="max", log_to_console=log_to_console, log_to_framework=log_to_framework
        )
        predictions = detach(predictions)
        labels = detach(labels)
        self.correct = (predictions == labels).sum().item()
        self.total = labels.size(0)

    @property
    def value(self):
        return self.correct / self.total

    def update(self, metric: Metric):
        self.correct += metric.correct
        self.total += metric.total


class LossMetric(RunningMeanMetric):
    base_tags = {"losses"}

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "loss",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best="min",
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )


class LLMetric(RunningMeanMetric):
    base_tags = {"log_likelihoods"}

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "ll",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best="max",
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )


class KLMetric(RunningMeanMetric):
    base_tags = {"kl_divergences"}

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "kl",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )


class BitsPerDimMetric(RunningMeanMetric):
    base_tags = set()
    _str_value_fmt = "<5.3"  # 5.321

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "bpd",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        values = -values / math.log(2)
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best="min",
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )


class PerplexityMetric(BitsPerDimMetric):
    """Perplexity computed as $2^{-\frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i)}$"""

    base_tags = set()
    _str_value_fmt = "<8.3"  # 8765.321

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "pp",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best="min",
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )

    @property
    def value(self):
        return 2 ** self._value
