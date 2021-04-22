import math

from copy import deepcopy
from typing import List, Optional, Set, Union
from vseq.utils.summary import get_number_of_elements

import torch

from vseq.utils.operations import detach


class Metric:
    base_tags = set()
    _str_value_fmt = "<10.3"

    def __init__(self, name: str, tags: set):
        self.name = name
        self.tags = self.base_tags if tags is None else (tags | self.base_tags)

    @property
    def value(self):
        """Primary value of the metric to be used for logging"""
        raise NotImplementedError()

    def update(self, metric):
        """Update the metric (e.g. running mean)"""
        raise NotImplementedError()

    @property
    def str_value(self):
        return f"{self.value:{self._str_value_fmt}f}"

    @staticmethod
    def get_best(metrics):
        """Return the best from list of metrics"""
        return None

    def copy(self):
        return deepcopy(self)


@staticmethod
def min_value(metrics: List[Metric]):
    return min(metrics, key=lambda m: m.value)


@staticmethod
def max_value(metrics: List[Metric]):
    return max(metrics, key=lambda m: m.value)


class RunningMeanMetric(Metric):
    _str_value_fmt = "<10.3"

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str,
        tags: Set[str],
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
    ):
        """Create a running mean metric.

        Args:
            values (Union[torch.Tensor, float]): Values of the metric
            name (str): Name of the metric
            tags (Set[str]): Tags to use for grouping with other metrics.
            reduce_by (Optional[Union[torch.Tensor, float]], optional): A single or per example divisor of the values. Defaults to batch size.
            weight_by (Optional[Union[torch.Tensor, float]], optional): A single or per example weights for the running mean. Defaults to `reduce_by`.
        """
        super().__init__(name=name, tags=tags)

        values = detach(values)
        reduce_by = detach(reduce_by)
        reduce_by = detach(reduce_by)

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
    _str_value_fmt = "<10.3"
    get_best = max_value

    def __init__(
        self,
        predictions: Union[torch.Tensor, float],
        labels: Union[torch.Tensor, float],
        name: str = "accuracy",
        tags: Set[str] = None,
    ):
        """Classification accuracy"""
        super().__init__(name, tags)
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
    get_best = min_value

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "loss",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
    ):
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by)


class LLMetric(RunningMeanMetric):
    base_tags = {"log_likelihoods"}
    get_best = max_value

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "ll",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
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
        weight_by: Optional[Union[torch.Tensor, float]] = None,
    ):
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by)


class BitsPerDimMetric(RunningMeanMetric):
    base_tags = set()
    get_best = min_value

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "bpd",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
    ):
        values = -values / math.log(2)
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by)


class PerplexityMetric(BitsPerDimMetric):
    """Perplexity computed as $2^{-\frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i)}$

    Args:
        RunningMeanMetric ([type]): [description]
    """

    base_tags = set()
    get_best = min_value

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str = "pp",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
    ):
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by)

    @property
    def value(self):
        return 2 ** self._value


class HoyerSparsityMetric(RunningMeanMetric):
    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str,
        tags: Set[str],
        reduce_by: Optional[Union[torch.Tensor, float]],
        weight_by: Optional[Union[torch.Tensor, float]],
        normalze: bool = True,
    ):
        """Sparsity of representation as computed in [1] and presented in [2].

        Args:
            values (Union[torch.Tensor, float]): [description]
            name (str): [description]
            tags (Set[str]): [description]
            reduce_by (Optional[Union[torch.Tensor, float]]): [description]
            weight_by (Optional[Union[torch.Tensor, float]]): [description]
            normalze (bool, optional): Normalize the values with the running standard deviation per dimension.
                                       This normalisation is important as one could achieve a “sparse” representation
                                       simply by having different dimensions vary along different length scales.
                                       Defaults to True.

        Raises:
            NotImplementedError: [description]

        [1] Hurley and Rickard, 2008
        [2] http://arxiv.org/abs/1812.02833 p. 7
        """
        super().__init__(values, name, tags, reduce_by=reduce_by, weight_by=weight_by)
        raise NotImplementedError()
