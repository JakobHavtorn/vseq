import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["WANDB_MODE"] = "disabled" # equivalent to "wandb disabled"

import argparse
import logging
import json

import torch
import wandb
import rich

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import vseq
import vseq.data
import vseq.models
import vseq.utils
import vseq.utils.device

from vseq.data import BaseDataset
from vseq.data.batchers import TextBatcher, SpectrogramBatcher, AudioBatcher
from vseq.data.datapaths import LIBRISPEECH_TRAIN, LIBRISPEECH_DEV_CLEAN
from vseq.data.loaders import AudioLoader
from vseq.data.token_map import TokenMap
from vseq.data.transforms import Compose, MuLawEncode
from vseq.data.samplers import LengthTrainSampler, LengthEvalSampler
from vseq.evaluation import Tracker
from vseq.training import set_dropout
from vseq.utils.rand import set_seed, get_random_seed
from vseq.models import DeepLSTMASR
from vseq.modules import STFTConv


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="batch size")
parser.add_argument("--max_second_diff", default=0.3, type=float, help="control the variation of sample lengths")
parser.add_argument("--sample_rate", default=16000, type=int, help="sample rate")
parser.add_argument("--n_fft", default=320, type=int, help="Number of FFTs")
parser.add_argument("--win_length", default=320, type=int, help="the size of the STFT window")
parser.add_argument("--hop_length", default=160, type=int, help="the stride of the STFT window")
parser.add_argument("--n_mels", default=80, type=int, help="number of mel filterbanks")

parser.add_argument("--lr_max", default=3e-4, type=float, help="start learning rate")
parser.add_argument("--lr_min", default=5e-5, type=float, help="end learning rate")
parser.add_argument("--optimizer", default='Adam', type=str, help="optimizer")
parser.add_argument("--optimizer_kwargs", default='{}', type=json.loads, help="extra kwargs for optimizer")

parser.add_argument("--layers_pr_block", default=5, type=int, help="number of LSTM layers pr block")
parser.add_argument("--hidden_size", default=320, type=int, help="size of the LSTM layers")
parser.add_argument("--dropout_prob", default=0.1, type=float, help="size of the LSTM layers")

parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
parser.add_argument("--warm_up", default=50, type=int, help="epochs before lr annealing starts")
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
    project="lstm-autoregressive",
    group=None,
)
wandb.config.update(args)

rich.print(vars(args))


audio_loader = AudioLoader("flac", cache=False, sum_channels=True)
audio_transform = MuLawEncode()
audio_batcher = AudioBatcher()

modalities = [
    (audio_loader, audio_transform, audio_batcher),
]

train_dataset = BaseDataset(
    source=LIBRISPEECH_TRAIN,
    modalities=modalities,
)
val_dataset = BaseDataset(
    source=LIBRISPEECH_DEV_CLEAN,
    modalities=modalities,
)

train_sampler = LengthTrainSampler(
    source=LIBRISPEECH_TRAIN,
    field="length.flac.samples",
    batch_size=args.batch_size,
    max_pool_difference=float(args.sample_rate * args.max_second_diff)
)

val_sampler = LengthEvalSampler(
    source=LIBRISPEECH_DEV_CLEAN,
    field="length.flac.samples",
    batch_size=args.batch_size
)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    batch_sampler=train_sampler
)

val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    batch_sampler=val_sampler
)

model = STFTConv() # dummy
model.to(device)

optimizer = getattr(torch.optim, args.optimizer)
optimizer = optimizer(model.parameters(), lr=args.lr_max, **args.optimizer_kwargs)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - args.warm_up), eta_min=args.lr_min)

tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    # update hyperparams (pre)
    p = min(epoch - 1, args.warm_up) / args.warm_up * args.dropout_prob
    set_dropout(model, p)

    # training
    model.train()
    for (x, x_sl), metadata in tracker.steps(train_loader):

        x = x.to(device)
        break
    break

    #     loss, metrics, outputs = model(x, x_sl, y, y_sl)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     tracker.update(metrics)

    # tracker.reset()

    # # evaluation
    # model.eval()
    # with torch.no_grad():
    #     for ((x, x_sl), (y, y_sl)), metadata in tracker.steps(val_loader):
            
    #         x = x.to(device)
    #         y = y.to(device)
    #         loss, metrics, outputs = model(x, x_sl, y, y_sl)

    #         tracker.update(metrics)

    # tracker.reset()

    # # update hyperparams (post)
    # if epoch >= args.warm_up:
    #     lr_scheduler.step()
