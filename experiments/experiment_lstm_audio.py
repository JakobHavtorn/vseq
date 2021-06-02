import os
import argparse
import logging

import torch
import torchaudio
import wandb
import rich

from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram, InverseMelScale, GriffinLim

import vseq.models

from vseq.data import BaseDataset
from vseq.data.batchers import AudioBatcher, SpectrogramBatcher
from vseq.data.datapaths import LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_TRAIN
from vseq.data.loaders import AudioLoader
from vseq.data.transforms import Compose, Quantize, RandomSegment, Scale, MuLawEncode
from vseq.evaluation.tracker import Tracker
from vseq.utils.argparsing import str2bool
from vseq.utils.device import get_device
from vseq.utils.rand import set_seed, get_random_seed
from vseq.models.lstm import LSTM


LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument(
    "--num_hidden",
    default=64,
    type=int,
    help="number of hidden units in LSTM",
)
parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
parser.add_argument(
    "--cache_dataset",
    default=False,
    type=str2bool,
    help="if True, cache the dataset in RAM",
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="number of dataloader workers"
)
parser.add_argument(
    "--wandb_group",
    default="lstm_audio",
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
parser.add_argument(
    "--save_freq", default=10, type=int, help="number of epochs to go between saves"
)
parser.add_argument(
    "--delete_last_model",
    default=False,
    type=str2bool,
    help="if True, delete the last model saved",
)
parser.add_argument(
    "--input_coding",
    default="stack",
    type=str,
    choices=["mu_law", "frames", "mel", "stack"],
    help="how to encode the input",
)

args, _ = parser.parse_known_args()

if args.seed is None:
    args.seed = get_random_seed()
set_seed(args.seed)

device = get_device() if args.device == "auto" else torch.device(args.device)


model_name_str = f"lstm_audio-{args.num_hidden}"
print(f"Initializing model with name: {model_name_str}")

wandb.init(
    entity="vseq",
    project="lstm",
    group=args.wandb_group,
)
wandb.config.update(args)
rich.print(vars(args))


_transforms = [
RandomSegment(length=16000) # (B, 16000)
]

if args.input_coding == "mu_law":
    _transforms.extend((MuLawEncode(), Quantize(bits=8)))
    batcher= AudioBatcher
elif args.input_coding == "frames":
    batcher = AudioBatcher
elif args.input_coding == "mel":
    _transforms.append(MelSpectrogram(n_mels=80))
    batcher = SpectrogramBatcher
elif args.input_coding == "stack":
    _transforms.extend([
        # Scale(min_val=-2**8, max_val=2**8), 
        torch.nn.Unflatten(0, (int(16000/80), 80)) # (B, T) -> (B, T/S, S) # S is pseudo-spectrogram dim  
    ])
    batcher = SpectrogramBatcher

data_transform = Compose(*_transforms)
rich.print(data_transform)
modalities = [
    (AudioLoader("flac"), data_transform, batcher()),
]

train_dataset = BaseDataset(
    source=LIBRISPEECH_TRAIN,
    modalities=modalities,
    sort=False,
)
val_dataset = BaseDataset(
    source=LIBRISPEECH_DEV_CLEAN,
    modalities=modalities,
    sort=False,
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


model = LSTM(input_size=80, hidden_size=args.num_hidden, num_classes=256)

model = model.to(device)
print(model)
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
        loss.sum().backward()
        optimizer.step()

        tracker.update(metrics)

    model.eval()
    with torch.no_grad():
        for (x, x_sl), metadata in tracker.steps(val_loader):
            x = x.to(device)

            loss, metrics, output = model(x, x_sl)

            tracker.update(metrics)

        tracker.log()

        # save samples
        # x = model.generate(n_samples=32, n_frames=96000)
        # x = x.unsqueeze(-1).to(torch.uint8).cpu()
        # for i in range(len(x)):
        #     torchaudio.save(
        #         f"./wavenet_samples/{model_name_str}-epoch-{epoch}-sample_{i}.wav",
        #         x[i],
        #         sample_rate=16000,
        #         channels_first=False,
        #         encoding="ULAW",
        #     )
    if epoch != 0 and epoch % args.save_freq == 0:  # epoch 10, 20, ...
        if args.delete_last_model and os.path.exists(
            f"./models/{model_name_str}-epoch-{epoch-args.save_freq}"
        ):
            # delete past model
            os.removedirs(f"./models/{model_name_str}-epoch-{epoch-args.save_freq}")
        model.save(f"./models/{model_name_str}-epoch-{epoch}")
