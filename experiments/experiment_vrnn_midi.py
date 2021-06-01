import argparse
import json

import torch
import wandb
import rich

from torch.utils.data import DataLoader

import vseq
import vseq.data
import vseq.models
import vseq.utils
import vseq.utils.device

from vseq.data import BaseDataset
from vseq.data.batchers import SpectrogramBatcher
from vseq.data.datapaths import MIDI_NOTTINGHAM_TEST, MIDI_NOTTINGHAM_TRAIN, MIDI_NOTTINGHAM_VALID
from vseq.data.loaders import NumpyLoader
from vseq.evaluation import Tracker
from vseq.utils.rand import set_seed, get_random_seed
from vseq.training.annealers import CosineAnnealer


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--hidden_size", default=512, type=int, help="dimensionality of hidden state in VRNN")
parser.add_argument("--latent_size", default=128, type=int, help="dimensionality of latent state in VRNN")
parser.add_argument("--dropout", default=0.0, type=float, help="inter GRU layer dropout probability")
parser.add_argument("--anneal_steps", default=5000, type=int, help="number of steps to anneal beta")
parser.add_argument("--anneal_start_value", default=0, type=float, help="initial beta annealing value")
parser.add_argument("--epochs", default=250, type=int, help="number of epochs")
parser.add_argument("--num_workers", default=4, type=int, help="number of dataloader workers")
parser.add_argument("--seed", default=None, type=int, help="random seed")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args, _ = parser.parse_known_args()


if args.seed is None:
    args.seed = get_random_seed()

set_seed(args.seed)

device = vseq.utils.device.get_device() if args.device == "auto" else torch.device(args.device)


wandb.init(
    entity="vseq",
    project="vrnn",
    group=None,
)
wandb.config.update(args)
rich.print(vars(args))

batcher = SpectrogramBatcher()
loader = NumpyLoader("npy", cache=True)

modalities = [(loader, None, batcher)]

train_dataset = BaseDataset(
    source=MIDI_NOTTINGHAM_TRAIN,
    modalities=modalities,
)
train_dataset.examples = train_dataset.examples * 10
val_dataset = BaseDataset(
    source=MIDI_NOTTINGHAM_VALID,
    modalities=modalities,
)
test_dataset = BaseDataset(
    source=MIDI_NOTTINGHAM_TEST,
    modalities=modalities,
)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    shuffle=True,
    batch_size=args.batch_size,
    pin_memory=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size,
    pin_memory=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    collate_fn=test_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size,
    pin_memory=True,
)


model = vseq.models.VRNN2D(
    input_size=88,
    hidden_size=args.hidden_size,
    latent_size=args.latent_size,
)
# model = vseq.models.lstm.LSTM2D(
#     input_size=88,
#     hidden_size=args.hidden_size,
# )

wandb.watch(model, log="all", log_freq=len(train_loader))
model = model.to(device)
print(model)
# x, x_sl = next(iter(train_loader))[0]
# x = x.to(device)
# model.summary(input_data=x, x_sl=x_sl)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

tracker = Tracker()

beta_annealer = CosineAnnealer(anneal_steps=args.anneal_steps, start_value=args.anneal_start_value, end_value=1)

for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker(train_loader):
        x = x.to(device)
        # rich.print(metadata)

        loss, metrics, outputs = model(x, x_sl, beta=beta_annealer.value)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)
        beta_annealer.step()

    model.eval()
    with torch.no_grad():
        for (x, x_sl), metadata in tracker(val_loader):
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl)

            tracker.update(metrics)

        for (x, x_sl), metadata in tracker(test_loader):
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl)

            tracker.update(metrics)

    tracker.log(beta=beta_annealer.value)
