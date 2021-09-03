import collections
import itertools
import psutil
import re

from blessed import Terminal
from collections import defaultdict
from datetime import datetime
from time import time
from types import SimpleNamespace
from typing import Dict, Iterable, Union, Any, List, Optional

import rich
import wandb
import torch.distributed as distributed

from torch.utils.data import DataLoader

from .metrics import Metric


FORMATTING_PATTERN = r"\[([^\]]+)\]"
FORMATTING_REGEX = re.compile(FORMATTING_PATTERN)


def length_of_formatting(string: str):
    return sum(len(s) + 2 for s in FORMATTING_REGEX.findall(string))  # plus 2 for parenthesis


def length_without_formatting(string: str):
    return len(string) - length_of_formatting(string)


def source_string(source):
    return f"{source[:18]}.." if len(source) > 20 else f"{source}"


def rank_string(rank):
    return f"[grey30]rank {rank:2d}[/grey30]"


class Tracker:
    def __init__(
        self,
        print_every: Union[int, float] = 1.0,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        cpu_util_window: int = 10,
    ) -> None:
        """Tracks metrics, prints to console and logs to wandb.

        Example using `.epochs()` and `.steps()`:
        ```
        for epoch in tracker.epochs(num_epochs):
            for (x, x_sl), metadata in tracker.steps(train_dataloader):
                ...
                tracker.update(metrics)

            for (x, x_sl), metadata in tracker.steps(valid_dataloader):
                ...
                tracker.update(metrics)

            tracker.log()
        ```

        Example without `.epochs()` and `.steps()`
        ```
        for epoch in range(num_epochs):
            tracker.set("train", max_steps=100)
            for (x, x_sl), metadata in range(train_dataloader):
                ...
                tracker.increment_step()
                tracker.update(metrics)
                tracker.print()
            tracker.unset()

            tracker.set("test", max_steps=10)
            for (x, x_sl), metadata in tracker.steps(valid_dataloader):
                ...
                tracker.increment_step()
                tracker.update(metrics)
                tracker.print()
            tracker.unset()

            tracker.log()
            tracker.reset()
        ```

        Args:
            min_indent (int): Minimum indent for dataset name. Defaults to 40.
            print_every (Union[int, float]): Time between prints measured in steps (if int) or seconds (if float).
                                             Defaults to 1.0 (seconds).
            rank (Optional[int]): Rank (index) of the current worker process if using Distributed Data Parallel (DDP).
            world_size (Optional[int]): Total number of worker processes if using Distributed Data Parallel (DDP).
        """

        self.print_every = print_every
        self.rank = 0 if rank is None else rank
        self.world_size = world_size

        # dynamic variables
        self.max_source_str_len = 0
        self.max_progress_str_len = 0

        # ddp logic
        assert (rank is None) == (world_size is None), "Must either set both `rank` and `world_size` or neither of them"
        self.is_ddp = rank is not None
        self.terminal = Terminal()

        # continously updated
        self.printed_last = 0
        self.log_line_len = 0
        self.cpu_utils = defaultdict(lambda: collections.deque(maxlen=cpu_util_window))
        self.iowait = "-"
        self.source = None
        self.start_time = defaultdict(lambda: None)
        self.end_time = defaultdict(lambda: None)
        self.epoch = 0
        self.step = defaultdict(lambda: 0)
        self.max_steps = defaultdict(lambda: 0)

        self.metrics = defaultdict(dict)  # dict(source=dict(metric.name=metric))
        self.accumulated_metrics = defaultdict(lambda: defaultdict(list))  # dict(source=dict(metric.name=list(metric)))

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

    def steps(
        self, steppable: Union[Iterable, DataLoader], source: Optional[str] = None, max_steps: Optional[int] = None
    ):
        if source is None and not isinstance(steppable, DataLoader):
            raise ValueError("Must provide `source` to .steps() if steppable is not a DataLoader")

        source = source if source is not None else steppable

        self.set(source, max_steps=max_steps)

        iterator = iter(steppable)

        if hasattr(iterator, "_workers"):
            workers = [psutil.Process(w.pid) for w in iterator._workers]
        else:
            workers = None

        for batch in iterator:
            yield batch
            self.increment_step()
            if self.do_print():
                self.print(workers=workers)

        self.unset()

    def increment_step(self):
        """Increment the internal step counter `self.step[self.source]`"""
        self.step[self.source] += 1

    def epochs(self, N) -> Iterable[int]:
        """Yields the epoch index while printing epoch number and epoch delimiter."""
        for epoch in range(1, N + 1):
            self.epoch = epoch

            if self.rank == 0:
                # print epoch and timestamp
                s = f"\n[bold bright_white]Epoch {epoch}:[/bold bright_white] "
                s += "[grey30]" + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "[/]"
                rich.print(s, flush=True)

            yield epoch

            if self.rank == 0:
                if self.is_ddp:
                    # print summary of gathered and reduced metrics
                    rich.print(f"[bold bright_white]Summary:[/bold bright_white] {' ' * (self.log_line_len - 18)}\n")
                    for source in self.best_values.keys():
                        self.print(source=source)
                        print(flush=True)

                print("-" * (self.log_line_len or 50), flush=True)

            self.reset()

    def __call__(self, loader: Union[str, DataLoader]):
        """Shortcut applicable to the standard case."""
        return self.steps(loader)

    def set(self, source: Union[str, DataLoader], max_steps: int = None):
        """Set source name, start time and maximum number of steps if available."""
        if isinstance(source, DataLoader):
            self.source = source.dataset.source
            self.max_steps[self.source] = len(source) if max_steps is None else max_steps
        else:
            self.source = source
            self.max_steps[self.source] = max_steps

        self.start_time[self.source] = time()

        if self.is_ddp and self.rank == 0:
            for rank in reversed(range(self.world_size)):
                rich.print(rank_string(rank) + " " + source_string(self.source) + " " * self.log_line_len, flush=True)
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
            ms_per_step = "-"
        else:
            duration = time() - self.start_time[source]
            ms_per_step = duration / self.step[source] * 1000
            ms_per_step = f"{int(ms_per_step):d}ms"
            mins = int(duration // 60)
            secs = int(duration % 60)
            duration = f"{mins:d}m {secs:d}s"

        if workers is not None:
            cpu = int(round(sum([p.cpu_percent(interval=0.0) for p in workers]), 0))  # percent since last call
            self.cpu_utils[self.rank].append(cpu)
            cpu_times = [p.cpu_times() for p in workers]  # accumulated over lifetime of process
            time_usr_sys = sum([sum(ct[:2]) for ct in cpu_times]) / len(workers)
            time_iowait = sum([ct.iowait for ct in cpu_times]) / len(workers)
            self.iowait = f"{time_usr_sys:.0f}/{time_iowait:.0f}"
        if len(self.cpu_utils[self.rank]):
            cpu = sum(self.cpu_utils[self.rank]) / len(self.cpu_utils[self.rank])
            cpu = f"{cpu:.0f}%"
        else:
            cpu = "-%"

        ps = f"{steps_frac} [bright_white not bold]({duration}, {ms_per_step}, {cpu} {self.iowait}s)[/]"  # +26 format

        # source string
        ss = source_string(source)

        # update dynamic string lengths
        self.max_source_str_len = max(self.max_source_str_len, len(ss))
        self.max_progress_str_len = max(self.max_progress_str_len, len(ps))

        # source progress string
        sp = f"{ss:<{self.max_source_str_len}} - {ps:<{self.max_progress_str_len}}"
        if self.is_ddp:
            sp = rank_string(self.rank) + " " + sp
            end = "\r" if self.rank == 0 else "\n"

        # metrics string
        sep = " [magenta]|[/] "  # " | "
        metrics = [f"{name} = {met.str_value}" for name, met in self.metrics[source].items() if met.log_to_console]
        if len(metrics) > 0:
            # maybe shorten metrics string to fit terminal
            metrics_len = [3 + len(m) for m in metrics]  # 3 is length of sep without formatting
            metrics_len[0] += 3  # add sep left of first metric
            metrics_len[-1] -= 1  # remove space right of sep right of last metric
            metrics_cumlen = list(itertools.accumulate(metrics_len))
            max_metrics_str_len = self.terminal.width - length_without_formatting(sp)
            if metrics_cumlen[-1] > max_metrics_str_len:
                # last before too long
                idx = next(i for i, v in enumerate(metrics_cumlen) if v > max_metrics_str_len - 3)
                metrics = metrics[:idx] + ["..."]
        ms = sep + sep.join(metrics)

        # final string
        s = f"{sp:<}{ms}"

        self.log_line_len = length_without_formatting(s)
        s = s + " " * 5  # add some whitespace to overwrite any lingering characters

        if self.is_ddp:
            # TODO Instead of rich, print with regular print and add colors manually.
            #      The current implementation has a race condition on placing the cursor
            #      and printing the line with rich. This is merged to one print call without rich.
            # print(self.terminal.move_y(self.terminal.height - self.rank - 2) + s, end='\r')#, flush=True)
            with self.terminal.location(0, self.terminal.height - 2 - self.rank):
                rich.print(s, end=end, flush=True)
        else:
            rich.print(s, end=end, flush=True)

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

    def update(self, metrics: List[Metric], source: Optional[str] = None):
        """Update all metrics tracked on `source` with the given `metrics` and add any not currently tracked"""
        names = [metric.name for metric in metrics]
        assert len(names) == len(set(names)), "Metrics must have unique names"
        source = self.source if source is None else source

        if self.start_time[source] is None:
            self.start_time[source] = time()

        for metric in metrics:
            if metric.name in self.metrics[source]:
                self.metrics[source][metric.name].update(metric)
            else:
                self.metrics[source][metric.name] = metric.copy()

    def ddp_gather_and_reduce(self, source):
        """Share metrics across all `Tracker` objects, reduce them in `rank==self.rank` and update the Tracker"""
        # gather various variables across processes
        namespace_to_gather = SimpleNamespace(
            rank=self.rank,
            metrics=self.metrics[source],
            max_source_str_len=self.max_source_str_len,
            max_progress_str_len=self.max_progress_str_len,
        )
        gathered_namespaces = [None] * self.world_size
        distributed.all_gather_object(gathered_namespaces, namespace_to_gather)

        # filter gathered metrics
        gathered_metrics = [ns.metrics for ns in gathered_namespaces if ns.rank != self.rank]

        # update metrics
        for metrics in gathered_metrics:
            m = list(metrics.values())
            self.update(m, source=source)

        # sync log line width
        self.max_source_str_len = max([ns.max_source_str_len for ns in gathered_namespaces])
        self.max_progress_str_len = max([ns.max_progress_str_len for ns in gathered_namespaces])

        # update steps to match total steps taken
        self.step[source] = self.step[source] * self.world_size
        self.max_steps[source] = self.max_steps[source] * self.world_size
