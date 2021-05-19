import argparse
import logging

import torch
import wandb
import rich

from torch.utils.data import DataLoader

import vseq.models

from vseq.data import BaseDataset
from vseq.data.batchers import AudioBatcher

# from vseq.data.datapaths import LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_TRAIN
from vseq.data.synthetic_data import SimpleSinusoidDataset
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
parser.add_argument(
    "--layer_size", default=2, type=int, help="number of layers per stack"
)
parser.add_argument("--stack_size", default=2, type=int, help="number of stacks")
parser.add_argument(
    "--res_channels",
    default=64,
    type=int,
    help="number of channels in residual connections",
)
parser.add_argument(
    "--input_coding",
    default="frames",
    type=str,
    choices=["mu_law", "frames"],
    help="how to encode the input",
)
parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
parser.add_argument(
    "--cache_dataset",
    default=True,
    type=str2bool,
    help="if True, cache the dataset in RAM",
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="number of dataloader workers"
)
parser.add_argument(
    "--wandb_group",
    default=None,
    type=str,
    help="custom group for this experiment (optional)",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="seed for random number generators. Random if -1.",
)
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args, _ = parser.parse_known_args()

if args.seed is None:
    args.seed = get_random_seed()
set_seed(args.seed)

device = get_device() if args.device == "auto" else torch.device(args.device)


if args.input_coding == "mu_law":
    args.in_channels = 256
elif args.input_coding == "frames":
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

a7frequencies = [
    220,  # a3
    277.18,  # c#4
    329.63,  # e4
    392.00,  # g4
    440,
]


train_dataset = SimpleSinusoidDataset(
    frequency=a7frequencies,
    cycle_amplitude=5,
)

val_dataset = SimpleSinusoidDataset(
    n_samples=64, frequency=a7frequencies, cycle_amplitude=5
)

train_loader = DataLoader(
    dataset=train_dataset,
    num_workers=args.num_workers,
    shuffle=True,
    batch_size=args.batch_size,
    # batch_sampler=train_sampler,
)
val_loader = DataLoader(
    dataset=val_dataset,
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
    out_classes=256,
)
model = model.to(device)
print(model)
rich.print(model.receptive_field)
wandb.watch(model, log="all", log_freq=len(train_loader))

# (x, x_sl) = next(iter(train_loader))
# x = x.to(device)
# print(model.summary(input_example=x, x_sl=x_sl))

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# x = model.generate(n_samples=1, n_frames=16000)
# torchaudio.save('./wavenet/sample.wav', x[0].cpu(), sample_rate=16000, channels_first=True, compression=None)


tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl) in tracker(train_loader):
        x = x.to(device)

        loss, metrics, output = model(x, x_sl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)

    model.eval()
    with torch.no_grad():
        for (x, x_sl) in tracker(val_loader):
            x = x.to(device)

            loss, metrics, output = model(x, x_sl)

            tracker.update(metrics)

        if epoch % 5 == 1:
            _gen = model.generate(n_samples=1, n_frames=16000)

            example_dict = {
                f"examples - epoch {epoch:03}": [
                    wandb.Audio(
                        x[0].cpu().numpy(), sample_rate=8000, caption="Synthetic Data"
                    ),
                    wandb.Audio(
                        ((output.logits.argmax(1)[0] - 128) / 128.0).cpu().numpy(),
                        sample_rate=8000,
                        caption="Reconstruction",
                    ),
                    wandb.Audio(
                        _gen.squeeze().cpu().numpy(),
                        sample_rate=8000,
                        caption="Generated",
                    ),
                ],
            }

            tracker.log(**example_dict)
        else:
            tracker.log()
