from datetime import datetime
from blessed import Terminal
from collections import defaultdict
from typing import Dict, Iterator, Union, Any, List, Optional
from time import time

import rich
import wandb
import torch.distributed as distributed

from torch.utils.data import DataLoader

from .metrics import Metric


def source_string(source):
    return f"{source[:18]}.." if len(source) > 20 else source

def rank_string(rank):
    return f"[grey30]rank {rank:2d} [/grey30]"


class Tracker:
    def __init__(
        self, min_indent: int = 35, print_every: Union[int, float] = 1.0, rank: int = None, world_size: int = None
    ) -> None:
        """Tracks metrics, prints to console and logs to wandb.

        Args:
            min_indent (int): Minimum indent for dataset name. Defaults to 35.
            print_every (Union[int, float]): Time between prints measured in steps (if int) or seconds (if float).
                                             Defaults to 1.0 (second).
        """

        self.min_indent = min_indent
        self.print_every = print_every
        self.rank = 0 if rank is None else rank
        self.world_size = world_size

        assert ((rank is None) == (world_size is None)), "Must set both `rank` and `world_size` or neither"
        self.is_ddp = (rank is not None)
        if self.is_ddp:
            self.terminal = Terminal()

        # continously updated
        self.printed_last = 0
        self.last_log_line_len = 0
        self.source = None
        self.max_steps = None
        self.start_time = defaultdict(lambda: None)
        self.end_time = defaultdict(lambda: None)
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

        self.start_time[self.source] = time()

        if self.is_ddp and self.rank == 0:
            for rank in reversed(range(self.world_size)):
                rich.print(rank_string(rank) + source_string(self.source) + ' ' * self.last_log_line_len, flush=True)
            rich.print(f'Running DDP with world_size={self.world_size}', flush=True, end='\r')

        if self.is_ddp:
            distributed.barrier()

    def reset(self):
        """Resets metric values and subset specific tracking values. Prints new line."""
        self.print(end="\n")
        if self.is_ddp:
            distributed.barrier()
            if self.rank == 0:  # TODO This can be removed if printing with `print` rather than rich
                print(self.terminal.move_xy(0, self.terminal.height - 2), flush=True)

        for name, metric in self.metrics[self.source].items():
            self.accumulated_metrics[self.source][name].append(metric.copy())

        self.end_time[self.source] = time()
        self.source = None
        self.max_steps = None
        self.printed_last = 0
        self.step = 0
        self.accumulated_output = defaultdict(list)

    def do_print(self) -> bool:
        """Print at first and last step and according to `print_every`"""
        t = time()
        if isinstance(self.print_every, float):
            do_print = (t - self.printed_last) > self.print_every
        else:
            do_print = (self.step % self.print_every) == 0 or self.step == 1

        if do_print:
            self.printed_last = t

        return do_print

    def print(self, end="\r", source: Optional[str] = None):
        """Print the current progress and metric values."""
        source = self.source if source is None else source

        # progress string
        steps_frac = f"{self.step}/-" if self.max_steps is None else f"{self.step}/{self.max_steps}"
        if self.start_time[source] is None:
            duration = "-"
        else:
            duration = time() - self.start_time[source]
            mins = int(duration // 60)
            secs = int(duration % 60)
            duration = f"{mins:d}m {secs:d}s"
        ps = f"{steps_frac} [not bold]([/not bold]{duration}[not bold])[/not bold]"  # +42 format

        # metrics string
        sep = "[magenta]|[/magenta]"  # +19 format pr metric
        ms = "".join([f"{sep} {metric.name} = {metric.str_value}" for metric in self.metrics[source].values()])

        # source string
        ss = source_string(source)

        # full log string
        sp = f"{ss} - {ps}"
        s = f"{sp:<{self.min_indent + 42}s}{ms}"

        if self.is_ddp:
            end = '\r' if self.rank == 0 else '\n'
            s = rank_string(self.rank) + s

            # TODO Instead of rich, print with regular print and add colors manually.
            #      The current implementation has a race condition on placing the cursor
            #      and printing the line with rich. This is merged to one print call without rich.
            with self.terminal.location(0, self.terminal.height - 2 - self.rank):
              rich.print(s, end=end, flush=True)
            # print(self.terminal.move_y(self.terminal.height - self.rank - 1) + s, end='\r')#, flush=True)
        else:
            rich.print(s, end=end, flush=True)

        self.last_log_line_len = len(s.strip()) - 42 - len(self.metrics[source]) * 19 

    def log(self, **extra_log_data: Dict[str, Any]):
        """Log all tracked metrics to experiment tracking framework and reset `metrics`."""

        # if ddp, gather and reduce metrics from other workers
        if self.is_ddp:
            metrics_per_rank = [None] * self.world_size
            distributed.all_gather_object(metrics_per_rank, (self.rank, self.metrics))
            if self.rank == 0:
                metrics_per_rank = [metrics for (rank, metrics) in metrics_per_rank if rank != 0]
                for metrics in metrics_per_rank:
                    for source in self.best_values.keys():
                        m = list(metrics[source].values())
                        self.update(m, source=source)
                self.step = 0

        if self.is_ddp and self.rank == 0:
            rich.print(f"[bold bright_white]Summary:[/bold bright_white] {' ' * (self.last_log_line_len - 18)}\n")
            for source in self.best_values.keys():
                self.print(source=source)
                print(flush=True)

        # add extra, best and tracker metrics
        values = self.values
        values.update(extra_log_data)
        for source in self.best_values.keys():
            values[source].update(self.best_values[source])
            values[source]["epoch_duration"] = self.end_time[source] - self.start_time[source]

        if self.rank == 0:
            wandb.log(values)

        self.metrics = defaultdict(dict)
        self.start_time = defaultdict(lambda: None)
        self.end_time = defaultdict(lambda: None)

    def update(self, metrics: List[Metric], source: Optional[str] = None):
        """Update all tracked metrics with the given metrics adding any not currently tracked"""
        names = [metric.name for metric in metrics]
        assert len(names) == len(set(names)), "Metrics must have unique names"
        source = self.source if source is None else source

        self.step += 1
        for metric in metrics:
            if metric.name in self.metrics[source]:
                self.metrics[source][metric.name].update(metric)
            else:
                self.metrics[source][metric.name] = metric.copy()

    def accumulate(self, **kwargs: Dict[Any, Any]):
        """Accumulate some outputs of interest. Gets reset on every `reset()` (e.g. epoch)"""
        for k, v in kwargs.items():
            self.accumulated_output[k].append(v)

    def epochs(self, N) -> Iterator[int]:
        """Yields the epoch index while printing epoch number and epoch delimiter."""
        for epoch in range(1, N + 1):
            self.epoch = epoch

            if self.rank == 0:
                s = f"\n[bold bright_white]Epoch {epoch}:[/bold bright_white] "
                s += datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                rich.print(s, flush=True)

            yield epoch

            if self.rank == 0:
                print("-" * (self.last_log_line_len or 50), flush=True)

    def steps(self, loader: Union[str, DataLoader]):
        return self(loader)
