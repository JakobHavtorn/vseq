from collections import defaultdict
from typing import Union, Any
from time import time

import rich
import wandb
import torch

from torch.utils.data import DataLoader

from .metrics import Metric


class Tracker:
    def __init__(self, min_indent: int = 35, device: torch.device = None) -> None:

        self.min_indent = min_indent
        self.device = device

        # continously updated
        self.last_log_line_len = None
        self.current_source = None
        self.max_steps = None
        self.start_time = None
        self.steps = 0
        self.sources = dict()

        self.accumulated_metrics = defaultdict(list)
        self.updated_metrics = dict()

    def __call__(self, loader):
        """Shortcut applicable to the standard case."""

        self.set(loader)
        for batch in loader:
            yield batch
            self.print()
        self.reset()

    def set(self, source: Union[str, DataLoader]):
        """Set source name, start time and maximum number of steps if available."""
        if isinstance(source, DataLoader):
            self.current_source = source.dataset.source
            self.max_steps = len(source)
        else:
            self.current_source = source
            self.max_steps = None

        self.start_time = time()

    def reset(self):
        """Resets metric values and subset specific tracking values. Prints new line."""

        print()

        log_values = {metric.name: metric.value for metric in self.updated_metrics.values()}
        self.sources[self.current_source] = log_values

        self.current_source = None
        self.max_steps = None
        self.start_time = None
        self.steps = 0
        self.accumulated_metrics = defaultdict(list)
        self.updated_metrics = dict()

    def print(self,  end='\r', max_steps=None):
        """Prints the current progress and metric values."""

        # progress string
        steps_frac = f"{self.steps}/-" if self.max_steps is None else f"{self.steps}/{self.max_steps}"
        duration = (time() - self.start_time)
        mins = int(duration // 60)
        secs = int(duration % 60)
        duration = "-" if self.start_time is None else f"{mins:d}m {secs:d}s"
        ps = f"{steps_frac} [not bold]([/not bold]{duration}[not bold])[/not bold]" # +42 format

        # metrics string
        sep = "[magenta]|[/magenta]" # +19 format pr metric
        ms = "".join([f"{sep} {metric.name} = {metric.str_value}" for metric in self.updated_metrics.values()])

        # source string
        ss = f"{self.current_source[:8]}.." if len(self.current_source) > 10 else self.current_source
        
        # full log string
        sp = f"{ss} - {ps}"
        s = f"{sp:<{self.min_indent + 42}s}{ms}"

        rich.print(s, end=end)
        self.last_log_line_len = len(s.strip()) - 42 - len(self.updated_metrics) * 19

    def log(self):
        """Logs epoch values to experimental tracking framework."""
        wandb.log(self.sources)
        self.sources = dict()

    def update(self, metrics: Metric):
        self.steps += 1
        for metric in metrics:
            if metric.name in self.updated_metrics:
                self.updated_metrics[metric.name].update(metric)  # metric.to(self.device)  # TODO Fix when device==None
            else:
                self.updated_metrics[metric.name] = metric

    def accumulate(self, **kwargs: Any):
        for k, v in kwargs.items():
            self.accumulated_metrics[k].append(v)

    def epochs(self, N):
        """Prints epoch number and  epoch delimiter."""

        for epoch in range(1, N + 1):
            rich.print(f"\n[bold bright_white]Epoch {epoch}:[/bold bright_white]")
            yield epoch
            print("-" * (self.last_log_line_len or 50))

