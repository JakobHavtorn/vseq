import argparse
import logging

import torch
import wandb
import rich

from torch.utils.data import DataLoader

import vseq.models

from vseq.data import BaseDataset
from vseq.data.batchers import AudioBatcher
from vseq.data.datapaths import LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_TRAIN
from vseq.data.transforms import Compose, Quantize, RandomSegment, Scale, MuLawEncode
from vseq.evaluation.tracker import Tracker
from vseq.utils.argparsing import str2bool
from vseq.utils.device import get_device
from vseq.utils.rand import set_seed, get_random_seed


torch.autograd.set_detect_anomaly(True)
LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--lr", default=2e-3, type=float, help="base learning rate")
parser.add_argument("--layer_size", default=2, type=int, help="number of layers per stack")
parser.add_argument("--stack_size", default=2, type=int, help="number of stacks")
parser.add_argument("--res_channels", default=64, type=int, help="number of channels in residual connections")
parser.add_argument("--input_coding", default='mu_law', type=str, choices=['mu_law', 'frames'], help="how to encode the input")
parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
parser.add_argument("--cache_dataset", default=True, type=str2bool, help="if True, cache the dataset in RAM")
parser.add_argument("--num_workers", default=8, type=int, help="number of dataloader workers")
parser.add_argument("--wandb_group", default=None, type=str, help='custom group for this experiment (optional)')
parser.add_argument("--seed", default=None, type=int, help="seed for random number generators. Random if -1.")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args, _ = parser.parse_known_args()

if args.seed is None:
    args.seed = get_random_seed()
set_seed(args.seed)

device = get_device() if args.device == "auto" else torch.device(args.device)


if args.input_coding == 'mu_law':
    args.in_channels = 256
elif args.input_coding == 'frames':
    args.in_channels = 1
else:
    raise ValueError()


wandb.init(
    entity="vseq",
    project="wavenet",
    group=None,
)
wandb.config.update(args)
rich.print(vars(args))


if args.input_coding == 'mu_law':
    wavenet_transform = Compose(RandomSegment(length=64000), MuLawEncode(), Quantize(bits=8))
elif args.input_coding == 'frames':
    wavenet_transform = Compose(RandomSegment(length=64000))

modalities = [
    ("flac", wavenet_transform, AudioBatcher()),
]

train_dataset = BaseDataset(
    source=LIBRISPEECH_TRAIN,
    modalities=modalities,
)
val_dataset = BaseDataset(
    source=LIBRISPEECH_DEV_CLEAN,
    modalities=modalities,
)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    shuffle=True,
    batch_size=4
    # batch_sampler=train_sampler,
)
val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=4
    # batch_sampler=val_sampler,
)

model = vseq.models.WaveNet(
    layer_size=args.layer_size,
    stack_size=args.stack_size,
    in_channels=args.in_channels,
    res_channels=args.res_channels,
    out_classes=256
)
model = model.to(device)
print(model)
rich.print(model.receptive_field)
wandb.watch(model, log='all', log_freq=len(train_loader))

(x, x_sl), metadata = next(iter(train_loader))
x = x.to(device)
print(model.summary(input_example=x, x_sl=x_sl))

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# x = model.generate(n_samples=1, n_frames=16000)
# torchaudio.save('./wavenet/sample.wav', x[0].cpu(), sample_rate=16000, channels_first=True, compression=None)


tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker(train_loader):
        x = x.to(device)

        loss, metrics, output = model(x, x_sl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)

    model.eval()
    for (x, x_sl), metadata in tracker(val_loader):
        x = x.to(device)

        loss, metrics, output = model(x, x_sl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)

    tracker.log()
