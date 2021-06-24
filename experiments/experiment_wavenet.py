import argparse
import logging

import torch
import torchaudio
import wandb
import rich

from torch.utils.data import DataLoader

import vseq.models

from vseq.data import BaseDataset
from vseq.data.batchers import AudioBatcher
from vseq.data.datapaths import TIMIT_TRAIN, TIMIT_TEST
from vseq.data.loaders import AudioLoader
from vseq.data.transforms import Compose, Quantize, RandomSegment, Scale, MuLawEncode, StackWaveform
from vseq.evaluation.tracker import Tracker
from vseq.utils.argparsing import str2bool
from vseq.utils.device import get_device
from vseq.utils.rand import set_seed, get_random_seed


LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--n_layers", default=10, type=int, help="number of layers per stack")
parser.add_argument("--n_stacks", default=4, type=int, help="number of stacks")
parser.add_argument("--res_channels", default=64, type=int, help="number of channels in residual connections")
parser.add_argument("--input_coding", default="mu_law", type=str, choices=["mu_law", "frames"], help="input encoding")
parser.add_argument("--num_bits", default=8, type=int, help="number of bits for mu_law encoding (note the data bits depth)")
parser.add_argument("--stack_frames", default=1, type=int, help="Number of audio frames to stack in feature vector if input_coding is frames")
parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
parser.add_argument("--cache_dataset", default=False, type=str2bool, help="if True, cache the dataset in RAM")
parser.add_argument("--num_workers", default=8, type=int, help="number of dataloader workers")
parser.add_argument("--wandb_group", default=None, type=str, help="custom group for this experiment (optional)")
parser.add_argument("--seed", default=None, type=int, help="seed for random number generators. Random if -1.")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args = parser.parse_args()

if args.seed is None:
    args.seed = get_random_seed()
set_seed(args.seed)

device = get_device() if args.device == "auto" else torch.device(args.device)


if args.input_coding == "frames":
    args.num_embeddings = None
elif args.input_coding == "mu_law":
    args.num_embeddings = 2 ** args.num_bits
else:
    raise ValueError()


wandb.init(
    entity="vseq",
    project="wavenet",
    group=None,
)
wandb.config.update(args)
rich.print(vars(args))


if args.input_coding == "mu_law":
    wavenet_transform = Compose(RandomSegment(length=16000), MuLawEncode(), Quantize(bits=args.num_bits))
elif args.input_coding == "frames":
    if args.stack_frames == 1:
        wavenet_transform = Compose(RandomSegment(length=16000))
    else:
        wavenet_transform = Compose(RandomSegment(length=16000), StackWaveform(n_frames=args.stack_frames))

modalities = [
    (AudioLoader("wav"), wavenet_transform, AudioBatcher()),
]

train_dataset = BaseDataset(
    source=TIMIT_TRAIN,
    modalities=modalities,
)
val_dataset = BaseDataset(
    source=TIMIT_TEST,
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

model = vseq.models.WaveNet(
    n_layers=args.n_layers,
    n_stacks=args.n_stacks,
    num_embeddings=args.num_embeddings,
    res_channels=args.res_channels,
    out_classes=256,
)
(x, x_sl), metadata = next(iter(train_loader))
model.summary(input_example=x, x_sl=x_sl)
model = model.to(device)
print(model)
rich.print(model.receptive_field)
wandb.watch(model, log="all", log_freq=len(train_loader))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


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
        for (x, x_sl), metadata in tracker.steps(val_loader):
            x = x.to(device)

            loss, metrics, output = model(x, x_sl)

            tracker.update(metrics)


        reconstructions = [wandb.Audio(output.x_hat[i].cpu().flatten().numpy(), caption=f"Reconstruction {i}", sample_rate=16000) for i in range(2)]

        x = model.generate(n_samples=2, n_frames=128000 // args.stack_frames)
        samples = [wandb.Audio(x[i].flatten().cpu().numpy(), caption=f"Sample {i}", sample_rate=16000) for i in range(2)]

        tracker.log(samples=samples, reconstructions=reconstructions)

        # for i in range(len(x)):
        #     torchaudio.save(
        #         f"./wavenet_samples/model-{args.n_layers}-{args.n_stacks}-{args.res_channels}-epoch-{epoch}-sample_{i}.wav",
        #         x[i],
        #         sample_rate=16000,
        #         channels_first=False,
        #         encoding="ULAW",
        #     )
