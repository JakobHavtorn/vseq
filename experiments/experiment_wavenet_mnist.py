import argparse
import logging

from functools import partial

import torch
import torchvision
import wandb
import rich
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import vseq
import vseq.models

from vseq.data import BaseDataset
from vseq.data.batchers import TensorBatcher
from vseq.data.transforms import Compose, Quantize, Scale, Reshape
from vseq.evaluation.tracker import Tracker
from vseq.utils.argparsing import str2bool
from vseq.utils.device import get_device
from vseq.utils.rand import set_seed, get_random_seed


torch.autograd.set_detect_anomaly(True)
LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--layer_size", default=7, type=int, help="number of layers per stack")
parser.add_argument("--stack_size", default=6, type=int, help="number of stacks")
parser.add_argument("--res_channels", default=64, type=int, help="number of channels in residual connections")
parser.add_argument("--input_coding", default="quantized", type=str, choices=["quantized", "frames"], help="input encoding")
parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
parser.add_argument("--cache_dataset", default=False, type=str2bool, help="if True, cache the dataset in RAM")
parser.add_argument("--num_workers", default=8, type=int, help="number of dataloader workers")
parser.add_argument("--wandb_group", default=None, type=str, help="custom group for this experiment (optional)")
parser.add_argument("--seed", default=None, type=int, help="seed for random number generators. Random if -1.")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args, _ = parser.parse_known_args()

if args.seed is None:
    args.seed = get_random_seed()
set_seed(args.seed)

device = get_device() if args.device == "auto" else torch.device(args.device)


if args.input_coding == "quantized":
    args.in_channels = 256
elif args.input_coding == "frames":
    args.in_channels = 1
else:
    raise ValueError()


wandb.init(
    entity="vseq",
    project="wavenet",
    group="mnist",
)
wandb.config.update(args)
rich.print(vars(args))


if args.input_coding == "quantized":
    wavenet_transform = Compose(torchvision.transforms.ToTensor(), Scale(-1, 1), Reshape(784), Quantize(bits=8))
elif args.input_coding == "frames":
    wavenet_transform = Compose(torchvision.transforms.ToTensor(), Scale(-1, 1), Reshape(784))

train_dataset = torchvision.datasets.MNIST(
    root=vseq.settings.DATA_DIRECTORY, train=True, transform=wavenet_transform, download=True
)
train_dataset.source = 'mnist_train'
train_dataset.batchers = [TensorBatcher()]
train_dataset.sort = False
train_dataset.num_modalities = 1
train_dataset.collate = partial(BaseDataset.collate, train_dataset)

valid_dataset = torchvision.datasets.MNIST(
    root=vseq.settings.DATA_DIRECTORY, train=False, transform=wavenet_transform, download=True
)
valid_dataset.source = 'mnist_val'
valid_dataset.batchers = [TensorBatcher()]
valid_dataset.sort = False
valid_dataset.num_modalities = 1
valid_dataset.collate = partial(BaseDataset.collate, valid_dataset)


train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    shuffle=True,
    batch_size=args.batch_size,
    pin_memory=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    collate_fn=valid_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size,
    pin_memory=True,
)

model = vseq.models.WaveNet(
    layer_size=args.layer_size,
    stack_size=args.stack_size,
    in_channels=args.in_channels,
    res_channels=args.res_channels,
    out_classes=256,
)
model = model.to(device)
print(model)
rich.print(model.receptive_field)
wandb.watch(model, log="all", log_freq=len(train_loader))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# (x, x_sl), metadata = next(iter(train_loader))
# x = x.to(device)
# print(model.summary(input_example=x, x_sl=x_sl))


tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker.steps(train_loader):
        x = x.to(device)

        loss, metrics, output = model(x, x_sl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)

    model.eval()
    with torch.no_grad():
        for (x, x_sl), metadata in tracker.steps(valid_loader):
            x = x.to(device)

            loss, metrics, output = model(x, x_sl)

            tracker.update(metrics)

        tracker.log()

        # save samples
        x = model.generate(n_samples=32, n_frames=784)
        x = x.view(32, 28, 28).cpu()
        for i in range(len(x)):
            plt.imshow(x[i])
            plt.savefig(f"./wavenet_samples/mnist-model-{args.layer_size}-{args.stack_size}-{args.res_channels}-epoch-{epoch}-sample_{i}.png")
