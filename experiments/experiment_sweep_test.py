"""A script to show how sweeps work and how early stopping can be used to save compute time and run more experiments"""

import argparse

from time import sleep

import numpy as np
import rich
import wandb
import uniplot

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--start_value", type=float, default=0, help="start value of the metric")
parser.add_argument("--mid_value", type=float, default=7, help="middle value of the metric")
parser.add_argument("--end_value", type=float, default=5, help="final value of the metric")
parser.add_argument("--noise_sigma", type=float, help="stddev of Gaussian noise level")
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
parser.add_argument("--duration", type=float, default=100, help="experiment duration in seconds")

args = parser.parse_args()

wandb.init(entity="vseq", project="sweep_test")
wandb.config.update(args)
rich.print("Configuration:")
rich.print(vars(args))

sleep_s = args.duration / args.epochs
midpoint = np.random.randint(0, args.epochs)

rich.print(f"Sleeping {sleep_s} s per epoch for {args.epochs} epoch for total duration of {args.duration} s.")
rich.print(f"Metric midpoint sampled from [0, {args.epochs}] and set to {midpoint} epochs.")

metric_start = np.linspace(args.start_value, args.mid_value, midpoint)
metric_end = np.linspace(args.mid_value, args.end_value, args.epochs - midpoint)
metric_values = np.concatenate([metric_start, metric_end])
metric_values += np.random.normal(loc=0, scale=args.noise_sigma, size=metric_values.shape)

best_metric_values = np.maximum.accumulate(metric_values)

rich.print("Sampled metric trajectory:")
uniplot.plot([metric_values, best_metric_values], title="metric", legend_labels=["metric", "best_metric"])

rich.print("Running experiment...")
for epoch in tqdm(range(args.epochs)):
    wandb.log(
        dict(
            metric=metric_values[epoch],
            best_metric=metric_values[: epoch + 1].max(),
            nested_metric=dict(metric2=np.random.randn()),
        )
    )
    sleep(sleep_s)

rich.print("Done!")
