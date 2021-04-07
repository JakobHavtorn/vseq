from typing import Iterable, Union
from types import SimpleNamespace
from time import time

import rich
import wandb

from torch.utils.data import Dataset, DataLoader

from .metrics import Metric


class Tracker:
    def __init__(self, *metrics: Iterable[Metric], min_indent: int = 35) -> None:
        self.metrics = metrics
        self.min_indent = min_indent

        # continiously updated
        self.last_log_line_len = None
        self.current_source = None
        self.max_steps = None
        self.start_time = None
        self.steps = 0
        self.sources = SimpleNamespace()

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

        # self.print(end='\n')
        print()
        log_values = SimpleNamespace(**{metric.name: metric.log_value for metric in self.metrics})
        # setattr(self, self.current_source, log_values)
        setattr(self.sources, self.current_source, log_values)
        for metric in self.metrics:
            metric.reset()
        
        self.current_source = None
        self.max_steps = None
        self.start_time = None
        self.steps = 0

    def print(self, end='\r', max_steps=None):
        """Prints the current progress and metric values."""

        # progress string
        self.steps += 1
        steps_frac = f"{self.steps}/-" if self.max_steps is None else f"{self.steps}/{self.max_steps}"
        duration = (time() - self.start_time)
        mins = int(duration // 60)
        secs = int(duration % 60)
        duration = "-" if self.start_time is None else f"{mins:d}m {secs:d}s"
        ps = f"{steps_frac} [not bold]([/not bold]{duration}[not bold])[/not bold]" # +42 format

        # metrics string
        sep = "[magenta]|[/magenta]" # +19 format pr metric
        ms = "".join([f"{sep} {metric.name} = {metric.str_value}" for metric in self.metrics])

        # full log string
        ss = f"{self.current_source} - {ps}"
        s = f"{ss:<{self.min_indent + 42}s}{ms}"

        rich.print(s, end=end)
        self.last_log_line_len = len(s.strip()) - 42 - len(self.metrics) * 19

    def epochs(self, N):
        """Prints epoch number and  epoch delimiter."""

        for epoch in range(1, N + 1):
            rich.print(f"\n[bold bright_white]Epoch {epoch}:[/bold bright_white]")
            yield epoch
            print("-" * (self.last_log_line_len or 50))

    def log(self):
        """Logs epoch values to experimental tracking framework."""
        data = {k: vars(v) for k, v in vars(self.sources).items()}
        wandb.log(data)
        self.sources = SimpleNamespace()

    def __call__(self, loader):
        """Shortcut applicable to the standard case."""

        self.set(loader)
        for batch in loader:
            yield batch
            self.print()
        self.reset()
