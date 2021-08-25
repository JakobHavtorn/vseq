import math

from copy import deepcopy
from typing import List, Optional, Set, Union

import torch

from vseq.utils.operations import detach, update_running_variance


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
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: str = None,
    ):
        """Create a latest mean metric that maintains the latest mean when updated.

        Args:
            values (Union[torch.Tensor, float]): Values of the metric
            name (str): Name of the metric
            tags (Set[str]): Tags to use for grouping with other metrics.
            reduce_by (Optional[Union[torch.Tensor, float]], optional): A single or per example divisor of the values. Defaults to `values.numel()`.
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
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: str = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        """Create a running mean metric that maintains the running mean when updated.

        Args:
            values (Union[torch.Tensor, float]): Values of the metric of shape (B,)
            name (str): Name of the metric
            tags (Set[str]): Tags to use for grouping with other metrics.
            reduce_by (Optional[Union[torch.Tensor, float]], optional): A single or per example divisor of the values. Defaults to `values.numel()`.
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
        self.running_mean = value / reduce_by

    @property
    def value(self):
        return self.running_mean

    def update(self, metric: Metric):
        """Update the running mean statistic.

        Args:
            metric (RunningMeanMetric): The running mean metric to update with
        """
        d = self.weight_by + metric.weight_by
        w1 = self.weight_by / d
        w2 = metric.weight_by / d

        self.running_mean = self.running_mean * w1 + metric.running_mean * w2
        self.weight_by = d


class RunningVarianceMetric(Metric):
    _str_value_fmt = "<.3"

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        name: str,
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: str = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        """Create a running variance metric that maintains the running variance when updated.

        Args:
            values (Union[torch.Tensor, float]): Values of the metric of shape (B,)
            name (str): Name of the metric
            tags (Set[str]): Tags to use for grouping with other metrics.
            reduce_by (Optional[Union[torch.Tensor, float]], optional): A single or per example divisor of the values. Defaults to `values.numel()`.
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

        # sum of squares of differences from the current mean
        self.weight_by = weight_by
        self.running_mean = value / reduce_by
        self.M2 = ((values - self.running_mean) ** 2).sum().item() if isinstance(values, torch.Tensor) else 0
        self.population_variance = self.M2 / (reduce_by - 1) if reduce_by > 1 else float("nan")  # unbiased variance

    @property
    def value(self):
        return self.population_variance

    def update(self, metric: Metric):
        """Update the running variance statistic.

        Args:
            metric (RunningMeanMetric): The running variance metric to update with.
        """
        var, avg, w, M2 = update_running_variance(
            avg_a=self.running_mean,
            avg_b=metric.running_mean,
            w_a=self.weight_by,
            w_b=metric.weight_by,
            M2_a=self.M2,
            M2_b=metric.M2,
        )
        self.running_mean = avg
        self.population_variance = var
        self.weight_by = w
        self.M2 = M2


class LatentActivityMetric(Metric):
    def __init__(
        self,
        values: Union[torch.Tensor, float],
        # seq_len: Union[torch.Tensor, int],
        # min_len: Optional[int] = None,
        threshold: Optional[float] = None,
        name: str = "latent_activity",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: str = "max",
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        """Create a latent activity metric that maintains a running variance over batch when updated.

        Args:
            values (Union[torch.Tensor, float]): Values of the metric of shape (B, D) or (B, T, D)
            name (str): Name of the metric
            tags (Set[str]): Tags to use for grouping with other metrics.
            reduce_by (Optional[Union[torch.Tensor, float]], optional): A single or per example divisor of the values. Defaults to `values.numel()`.
            weight_by (Optional[Union[torch.Tensor, float]], optional): A single or per example weights for the running mean. Defaults to `reduce_by`.
        """
        assert values.ndim == 2 or values.ndim == 3, "latents must have shape (B, D) or (B, T, D)"

        super().__init__(
            name=name, tags=tags, get_best=get_best, log_to_console=log_to_console, log_to_framework=log_to_framework
        )

        self.threshold = threshold
        self.is_temporal = values.ndim == 3

        self._str_value_fmt = "<.3" if threshold is None else "<5.1"  # raw variance or percent

        values = detach(values)
        reduce_by = detach(reduce_by)
        weight_by = detach(weight_by)

        numel = values.size(0) if isinstance(values, torch.Tensor) else 1
        values = values.cpu() if isinstance(values, torch.Tensor) else values

        reduce_by = reduce_by.sum().tolist() if isinstance(reduce_by, torch.Tensor) else (reduce_by or numel)
        weight_by = weight_by.sum().tolist() if isinstance(weight_by, torch.Tensor) else (weight_by or reduce_by)

        self.weight_by = weight_by
        self.running_mean = values.mean(0)  # mean over batch (T, D)
        self.M2 = ((values - self.running_mean) ** 2).sum(0) if isinstance(values, torch.Tensor) else 0  # (T, D)

        # report average variance if threshold is None, otherwise percentage with variance over threshold
        self.running_variance = self.M2 / (reduce_by - 1) if reduce_by > 1 else float("nan")

        if reduce_by > 1 and self.threshold is not None:
            self.activity = (self.running_variance > self.threshold).sum() / self.running_variance.numel() * 100
        elif reduce_by > 1 and self.threshold is None:
            self.activity = self.running_variance.mean()
        else:
            self.activity = float("nan")

    @property
    def value(self):
        return self.activity

    def update(self, metric: Metric):
        """Update the running variance statistic.

        Args:
            metric (RunningMeanMetric): The running variance metric to update with.
        """
        if self.is_temporal:
            # cut off to shortest
            T = min(self.running_mean.size(0), metric.running_mean.size(0))
            self.running_mean = self.running_mean[:T]
            self.M2 = self.M2[:T]
            metric.running_mean = metric.running_mean[:T]
            metric.M2 = metric.M2[:T]

        var, avg, w, M2 = update_running_variance(
            avg_a=self.running_mean,
            avg_b=metric.running_mean,
            w_a=self.weight_by,
            w_b=metric.weight_by,
            M2_a=self.M2,
            M2_b=metric.M2,
        )
        if self.threshold is not None:
            activity = (var > self.threshold).sum() / var.numel() * 100
        else:
            activity = var.mean()

        self.activity = activity
        self.running_mean = avg
        self.weight_by = w
        self.M2 = M2


class AccuracyMetric(Metric):
    _str_value_fmt = "6.4"  # 6.4321

    def __init__(
        self,
        predictions: Union[torch.Tensor, float],
        labels: Union[torch.Tensor, float],
        name: str = "accuracy",
        tags: Set[str] = None,
        get_best: float = "min",
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        """Standard classification accuracy"""
        super().__init__(
            name=name, tags=tags, get_best=get_best, log_to_console=log_to_console, log_to_framework=log_to_framework
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
        get_best: float = "min",
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best=get_best,
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
        get_best: float = "max",
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best=get_best,
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
        get_best: float = None,
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best=get_best,
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
        get_best: float = "min",
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        values = -detach(values) / math.log(2)
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best=get_best,
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )


class ScaledBitsPerDimMetric(BitsPerDimMetric):
    """ScaledBitsPerDimMetric computed as $\frac{-\frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i)}{K}$ 
    with K as the scaling factor"""

    base_tags = set()
    get_best = min_value
    _str_value_fmt = "<5.3"  # 5.321

    def __init__(
        self,
        values: Union[torch.Tensor, float],
        num : int = 1,
        name: str = "sbpd",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
    ):
        values = -values / (num * math.log(2))
        super().__init__(values=values, name=name, tags=tags, reduce_by=reduce_by, weight_by=weight_by)


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
        get_best: float = "min",
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best=get_best,
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )

    @property
    def value(self):
        return 2 ** self.running_mean


class HoyerSparsityMetric(RunningMeanMetric):
    def __init__(
        self,
        values: torch.Tensor,
        name: str = "hoyer sparsity",
        tags: Set[str] = None,
        reduce_by: Optional[Union[torch.Tensor, float]] = None,
        weight_by: Optional[Union[torch.Tensor, float]] = None,
        get_best: float = "max",
        log_to_console: bool = True,
        log_to_framework: bool = True,
    ):
        """Sparsity of representation as computed in [1] and presented in [2].

        Args:
            values (torch.Tensor): Tensor of shape (*, D) where sparsity is computed over D and averaged over *.
            name (str): [description]
            tags (Set[str]): [description]
            reduce_by (Optional[Union[torch.Tensor, float]]): [description]
            weight_by (Optional[Union[torch.Tensor, float]]): [description]
            normalze (bool, optional): Normalize the values with the running standard deviation per dimension.
                                       This normalisation is important as one could achieve a “sparse” representation
                                       simply by having different dimensions vary along different length scales.
                                       Defaults to True.

        [1] Hurley and Rickard, 2008
        [2] http://arxiv.org/abs/1812.02833 p. 7
        """
        values = self.compute_sparsity(detach(values))
        super().__init__(
            values=values,
            name=name,
            tags=tags,
            reduce_by=reduce_by,
            weight_by=weight_by,
            get_best=get_best,
            log_to_console=log_to_console,
            log_to_framework=log_to_framework,
        )

    def compute_sparsity(self, values: torch.Tensor) -> torch.Tensor:
        D = values.size(-1)
        sqrt_d = math.sqrt(D)
        l1 = torch.linalg.norm(values, ord=1, dim=-1)
        l2 = torch.linalg.norm(values, ord=2, dim=-1)
        hoyer_sparsity = (sqrt_d - l1 / l2) / (sqrt_d - 1)
        return hoyer_sparsity
