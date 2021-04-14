from collections import defaultdict
from typing import Dict, Iterator, Union, Any
from time import time

import rich
import wandb

from torch.utils.data import DataLoader

from .metrics import Metric


class Tracker:
    def __init__(self, min_indent: int = 35, print_every: Union[int, float] = 1.0) -> None:
        """Tracks metrics, prints to console and logs to wandb.

        Args:
            min_indent (int): Minimum indent for dataset name. Defaults to 35.
            print_every (Union[int, float]): Time between prints measured in steps (if int) or seconds (if float).
                                             Defaults to 1.0 (second).
        """

        self.min_indent = min_indent
        self.print_every = print_every

        # continously updated
        self.printed_last = 0
        self.last_log_line_len = None
        self.source = None
        self.max_steps = None
        self.start_time = None
        self.step = 0
        self.epoch = 0

        self.metrics = defaultdict(dict)  # dict(source=dict(metric.name=metric))
        self.accumulated_metrics = defaultdict(lambda: defaultdict(list))  # dict(source=dict(metric.name=[metric]))
        self.accumulated_output = defaultdict(list)  # dict(output.name=[output.value])

    @property
    def values(self) -> Dict[str, Dict[str, float]]:
        """Values of metrics as nested dict"""
        return {
            source: {metric.name: metric.value for metric in self.metrics[source].values()}
            for source in self.metrics.keys()
        }

    @property
    def accumulated_values(self) -> Dict[str, Dict[str, Metric]]:
        """Accumulated values of metrics as nested dict"""
        return {
            source: {
                metrics[0].name: [metric.value for metric in metrics]
                for metrics in self.accumulated_metrics[source].values()
            }
            for source in self.accumulated_metrics.keys()
        }

    @property
    def best_metrics(self) -> Dict[str, Dict[str, Metric]]:
        """Best metrics according to `metric.get_best` as nested dict"""
        best = dict()
        for source in self.accumulated_metrics.keys():
            best[source] = dict()
            for name, acc_metrics in self.accumulated_metrics[source].items():
                metric = acc_metrics[0].get_best(acc_metrics)
                if metric is not None:
                    best[source][f"best_{name}"] = metric
        return best

    @property
    def best_values(self) -> Dict[str, Dict[str, float]]:
        """Values of the best metrics according to `metric.get_best` as nested dict"""
        best_metrics = self.best_metrics
        return {
            source: {name: metric.value for name, metric in best_metrics[source].items()}
            for source in best_metrics.keys()
        }

    def __call__(self, loader: Union[str, DataLoader]):
        """Shortcut applicable to the standard case."""
        self.set(loader)
        for batch in loader:
            yield batch
            if self.do_print():
                self.print()
        self.reset()

    def set(self, source: Union[str, DataLoader]):
        """Set source name, start time and maximum number of steps if available."""
        if isinstance(source, DataLoader):
            self.source = source.dataset.source
            self.max_steps = len(source)
        else:
            self.source = source
            self.max_steps = None

        self.start_time = time()

    def reset(self):
        """Resets metric values and subset specific tracking values. Prints new line."""
        self.print(end="\n")

        for name, metric in self.metrics[self.source].items():
            self.accumulated_metrics[self.source][name].append(metric.copy())

        self.source = None
        self.max_steps = None
        self.start_time = None
        self.step = 0
        self.accumulated_output = defaultdict(list)

    def do_print(self) -> bool:
        """Print at first and last step and according to `print_every`"""
        t = time()

        if isinstance(self.print_every, float):
            do_print = (t - self.printed_last) > 1
        else:
            do_print = (self.step % self.print_every) == 0 or self.step == 1

        if do_print:
            self.printed_last = t

        return do_print

    def print(self, end="\r"):
        """Print the current progress and metric values."""
        # progress string
        steps_frac = f"{self.step}/-" if self.max_steps is None else f"{self.step}/{self.max_steps}"
        duration = time() - self.start_time
        mins = int(duration // 60)
        secs = int(duration % 60)
        duration = "-" if self.start_time is None else f"{mins:d}m {secs:d}s"
        ps = f"{steps_frac} [not bold]([/not bold]{duration}[not bold])[/not bold]"  # +42 format

        # metrics string
        sep = "[magenta]|[/magenta]"  # +19 format pr metric
        ms = "".join([f"{sep} {metric.name} = {metric.str_value}" for metric in self.metrics[self.source].values()])

        # source string
        ss = f"{self.source[:8]}.." if len(self.source) > 10 else self.source

        # full log string
        sp = f"{ss} - {ps}"
        s = f"{sp:<{self.min_indent + 42}s}{ms}"

        rich.print(s, end=end)
        self.last_log_line_len = len(s.strip()) - 42 - len(self.metrics[self.source]) * 19

    def log(self, **extra_log_data):
        """Log all tracked metrics to experiment tracking framework and reset `metrics`."""
        values = self.values
        values.update(extra_log_data)
        for source in self.best_values.keys():
            values[source].update(self.best_values[source])

        wandb.log(values)
        self.metrics = defaultdict(dict)

    def update(self, metrics: Metric):
        """Update all tracked metrics with the given metrics adding any not currently tracked"""
        names = [metric.name for metric in metrics]
        assert len(names) == len(set(names)), "Metrics must have unique names"

        self.step += 1
        for metric in metrics:
            if metric.name in self.metrics[self.source]:
                self.metrics[self.source][metric.name].update(metric)
            else:
                self.metrics[self.source][metric.name] = metric.copy()

    def accumulate(self, **kwargs: Dict[Any, Any]):
        """Accumulate some outputs of interest. Gets reset on every `reset()` (e.g. epoch)"""
        for k, v in kwargs.items():
            self.accumulated_output[k].append(v)

    def epochs(self, N) -> Iterator[int]:
        """Yields the epoch index while printing epoch number and epoch delimiter."""
        for epoch in range(1, N + 1):
            self.epoch = epoch
            rich.print(f"\n[bold bright_white]Epoch {epoch}:[/bold bright_white]")
            yield epoch
            print("-" * (self.last_log_line_len or 50))
