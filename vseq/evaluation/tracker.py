import collections
import re
import psutil

from datetime import datetime
from blessed import Terminal
from collections import defaultdict
from typing import Dict, Iterable, Union, Any, List, Optional
from time import time

import rich
import wandb
import torch.distributed as distributed

from torch.utils.data import DataLoader

from .metrics import Metric

FORMATTING_PATTERN = r"\[([^\]]+)\]"
FORMATTING_REGEX = re.compile(FORMATTING_PATTERN)


def length_of_formatting(string: str):
    return sum(len(s) + 2 for s in FORMATTING_REGEX.findall(string))


def length_without_formatting(string: str):
    return len(string) - length_of_formatting(string)


def source_string(source):
    return f"{source[:18]}.." if len(source) > 20 else f"{source}"


def rank_string(rank):
    return f"[grey30]rank {rank:2d}[/grey30]"


class Tracker:
    def __init__(
        self,
        min_indent: int = 40,
        print_every: Union[int, float] = 1.0,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        cpu_util_window: int = 10
    ) -> None:
        """Tracks metrics, prints to console and logs to wandb.

        Args:
            min_indent (int): Minimum indent for dataset name. Defaults to 40.
            print_every (Union[int, float]): Time between prints measured in steps (if int) or seconds (if float).
                                             Defaults to 1.0 (seconds).
            rank (Optional[int]): Rank (index) of the current worker process if using Distributed Data Parallel (DDP).
            world_size (Optional[int]): Total number of worker processes if using Distributed Data Parallel (DDP).
        """

        self.min_indent = min_indent  # TODO Compute `min_indent` dynamically
        self.print_every = print_every
        self.rank = 0 if rank is None else rank
        self.world_size = world_size

        # ddp logic
        assert (rank is None) == (world_size is None), "Must either set both `rank` and `world_size` or neither of them"
        self.is_ddp = rank is not None
        self.terminal = Terminal()

        # continously updated
        self.printed_last = 0
        self.last_log_line_len = 0
        self.cpu_utils = defaultdict(lambda: collections.deque(maxlen=cpu_util_window))
        self.source = None
        self.start_time = defaultdict(lambda: None)
        self.end_time = defaultdict(lambda: None)
        self.epoch = 0
        self.step = defaultdict(lambda: 0)
        self.max_steps = defaultdict(lambda: 0)

        self.metrics = defaultdict(dict)  # dict(source=dict(metric.name=metric))
        self.accumulated_metrics = defaultdict(lambda: defaultdict(list))  # dict(source=dict(metric.name=list(metric)))
        self.accumulated_output = defaultdict(list)  # dict(key=list(value))

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

    def steps(self, steppable: Union[Iterable, DataLoader], source: Optional[str] = None):
        if source is None and not isinstance(steppable, DataLoader):
            raise ValueError("Must call steps() on a DataLoader if source is None")

        source = source if source is not None else steppable

        self.set(source)
        iterator = iter(steppable)

        if hasattr(iterator, "_workers"):
            workers = [psutil.Process(w.pid) for w in iterator._workers]
        else:
            workers = None

        for batch in iterator:
            yield batch
            if self.do_print():
                self.print(workers=workers)

        self.unset()

    def epochs(self, N) -> Iterable[int]:
        """Yields the epoch index while printing epoch number and epoch delimiter."""
        for epoch in range(1, N + 1):
            self.epoch = epoch

            if self.rank == 0:
                s = f"\n[bold bright_white]Epoch {epoch}:[/bold bright_white] "
                s += "[grey30]" + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "[/]"
                rich.print(s, flush=True)

            yield epoch

            if self.rank == 0:
                print("-" * (self.last_log_line_len or 50), flush=True)

    def __call__(self, loader: Union[str, DataLoader]):
        """Shortcut applicable to the standard case."""
        return self.steps(loader)

    def set(self, source: Union[str, DataLoader]):
        """Set source name, start time and maximum number of steps if available."""
        if isinstance(source, DataLoader):
            self.source = source.dataset.source
            self.max_steps[self.source] = len(source)
        else:
            self.source = source
            self.max_steps[self.source] = None

        self.start_time[self.source] = time()

        if self.is_ddp and self.rank == 0:
            for rank in reversed(range(self.world_size)):
                rich.print(
                    rank_string(rank) + " " + source_string(self.source) + " " * self.last_log_line_len, flush=True
                )
            rich.print(f"Running DDP with world_size={self.world_size}", flush=True, end="\r")

        if self.is_ddp:
            distributed.barrier()

    def unset(self):
        """Resets print timer and prints final line, unsets `source` and accumulates metrics."""
        self.print(end="\n")  # print line for last iteration regardless of `do_print()`

        if self.is_ddp:
            self.ddp_gather_and_reduce(self.source)

            # reset terminal to bottom of terminal window
            # NOTE This is due to risk of race condition not resetting this properly.
            # TODO Can be removed if printing with `print` rather than rich.
            if self.rank == 0:
                print(self.terminal.move_xy(0, self.terminal.height - 2), flush=True)

        self.end_time[self.source] = time()

        for name, metric in self.metrics[self.source].items():
            self.accumulated_metrics[self.source][name].append(metric.copy())

        self.source = None
        self.printed_last = 0
        self.accumulated_output = defaultdict(list)
        self.cpu_utils[self.rank] = collections.deque(maxlen=10)

    def reset(self):
        """Reset all per-source attributes"""
        self.metrics = defaultdict(dict)
        self.start_time = defaultdict(lambda: None)
        self.end_time = defaultdict(lambda: None)
        self.step = defaultdict(lambda: 0)
        self.max_steps = defaultdict(lambda: 0)

    def do_print(self) -> bool:
        """Print at first and last step and according to `print_every`"""
        t = time()
        if isinstance(self.print_every, float):
            do_print = (t - self.printed_last) > self.print_every
        else:
            do_print = (self.step[self.source] % self.print_every) == 0 or self.step[self.source] == 1

        if do_print:
            self.printed_last = t

        return do_print

    def print(self, end="\r", source: Optional[str] = None, workers: list = None):
        """Print the current progress and metric values."""
        source = self.source if source is None else source

        # progress string
        if self.max_steps[source]:
            steps_frac = f"{self.step[source]}/{self.max_steps[source]}"
        else:
            steps_frac = f"{self.step[source]}/-"

        ps = f"{steps_frac}"
        if self.start_time[source] is None:
            duration = "-"
            s_per_step = "-"
        else:
            duration = time() - self.start_time[source]
            s_per_step = duration / self.step[source] * 1000
            s_per_step = f"{int(s_per_step):d}ms"
            mins = int(duration // 60)
            secs = int(duration % 60)
            duration = f"{mins:d}m {secs:d}s"

        if workers is not None:
            cpu = int(round(sum([p.cpu_percent(interval=0.0) for p in workers]), 0))
            self.cpu_utils[self.rank].append(cpu)
        if len(self.cpu_utils[self.rank]):
            cpu = sum(self.cpu_utils[self.rank]) / len(self.cpu_utils[self.rank])
            cpu = f"{cpu:.0f}%"
        else:
            cpu = "-"

        ps = f"{steps_frac} [bright_white not bold]({duration}, {s_per_step}, {cpu})[/]"  # +26 format

        # source string
        ss = source_string(source)

        # metrics string
        sep = " [magenta]|[/]"  # +19 format pr metric
        ms = (
            "".join(
                [
                    f"{sep} {metric.name} = {metric.str_value}"
                    for metric in self.metrics[source].values()
                    if metric.log_to_console
                ]
            )
            + sep
        )

        # full log string
        sp = f"{ss} - {ps}"
        s = f"{sp:<{self.min_indent + length_of_formatting(sp)}s}{ms}"

        if self.is_ddp:
            # TODO Instead of rich, print with regular print and add colors manually.
            #      The current implementation has a race condition on placing the cursor
            #      and printing the line with rich. This is merged to one print call without rich.
            # print(self.terminal.move_y(self.terminal.height - self.rank - 2) + s, end='\r')#, flush=True)
            end = "\r" if self.rank == 0 else "\n"
            s = rank_string(self.rank) + " " + s
            s_len = length_without_formatting(s)
            s = s + " " * (self.terminal.width - s_len)
            with self.terminal.location(0, self.terminal.height - 2 - self.rank):
                rich.print(s, end=end, flush=True)
        else:
            s_len = length_without_formatting(s)
            s = s + " " * (self.terminal.width - s_len)
            rich.print(s, end=end, flush=True)

        self.last_log_line_len = s_len

    def log(self, **extra_log_data: Dict[str, Any]):
        """Log all tracked metrics to experiment tracking framework and reset `metrics`."""
        # add best and tracker metrics and any `extra_log_data`
        values = self.values
        values.update(extra_log_data)
        for source in self.best_values.keys():
            values[source].update(self.best_values[source])
            values[source]["epoch_duration"] = self.end_time[source] - self.start_time[source]

        if self.rank == 0:
            wandb.log(values)

        self.reset()

    def update(self, metrics: List[Metric], source: Optional[str] = None):
        """Update all metrics tracked on `source` with the given `metrics` and add any not currently tracked"""
        names = [metric.name for metric in metrics]
        assert len(names) == len(set(names)), "Metrics must have unique names"
        source = self.source if source is None else source

        if self.start_time[source] is None:
            self.start_time[source] = time()

        self.step[self.source] += 1

        for metric in metrics:
            if metric.name in self.metrics[source]:
                self.metrics[source][metric.name].update(metric)
            else:
                self.metrics[source][metric.name] = metric.copy()

    def accumulate(self, **kwargs: Dict[Any, Any]):
        """Accumulate some outputs of interest. Gets reset on every call to `reset()` (e.g. epoch)"""
        for k, v in kwargs.items():
            self.accumulated_output[k].append(v)

    def ddp_gather_and_reduce(self, source):
        """Share metrics across all `Tracker` objects, reduce them in `rank==self.rank` and update the Tracker"""
        # gather metric objects across processes
        metrics_per_rank = [None] * self.world_size
        distributed.all_gather_object(metrics_per_rank, (self.rank, self.metrics[source]))

        # filter gathered metrics
        gathered_metrics = [metrics for (rank, metrics) in metrics_per_rank if rank != self.rank]

        # update metrics
        for metrics in gathered_metrics:
            m = list(metrics.values())
            self.update(m, source=source)

        # update steps to match total steps taken
        self.step[source] = self.step[source] * (len(gathered_metrics) + 1)
        self.max_steps[source] = self.max_steps[source] * (len(gathered_metrics) + 1)

        if self.rank == 0:
            # print summary of gathered and redued metrics
            rich.print(f"[bold bright_white]Summary:[/bold bright_white] {' ' * (self.last_log_line_len - 18)}\n")
            for source in self.best_values.keys():
                self.print(source=source)
                print(flush=True)

        distributed.barrier()
