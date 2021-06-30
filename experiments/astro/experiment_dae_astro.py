import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
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
from vseq.data.batchers import TextBatcher, AudioBatcher
from vseq.data.datapaths import LIBRISPEECH_TRAIN, LIBRISPEECH_DEV_CLEAN
from vseq.data.tokens import ENGLISH_STANDARD, BLANK_TOKEN
from vseq.data.tokenizers import char_tokenizer
from vseq.data.loaders import TextLoader
from vseq.data.token_map import TokenMap
from vseq.data.transforms import Compose, EncodeInteger, TextCleaner, AstroSpeech
from vseq.data.samplers import LengthTrainSampler, LengthEvalSampler
from vseq.evaluation import Tracker
from vseq.training import set_dropout
from vseq.utils.rand import set_seed, get_random_seed
from vseq.models import AstroDAE


parser = argparse.ArgumentParser()
parser.add_argument("--seconds_pr_batch", default=160, type=int, help="batch size")
parser.add_argument("--max_second_diff", default=0.3, type=float, help="control the variation of sample lengths")
parser.add_argument("--sample_rate", default=8000, type=int, help="sample rate")

parser.add_argument("--duration", default=500, type=int, help="number of frames for each astro speech token")
parser.add_argument("--fade", default=150, type=int, help="cos-annealing duration for fading in/out astro tokens")
parser.add_argument("--mel_delta", default=60, type=int, help="the mel-frequency shift between tokens")
parser.add_argument("--token_shift", default=0, type=int, help="frequency span for each astro token")

parser.add_argument("--lr_max", default=3e-4, type=float, help="start learning rate")
parser.add_argument("--lr_min", default=5e-5, type=float, help="end learning rate")
parser.add_argument("--optimizer", default='Adam', type=str, help="optimizer")
parser.add_argument("--optimizer_kwargs", default='{}', type=json.loads, help="extra kwargs for optimizer")

parser.add_argument("--conv_kernels", default=[10, 10, 5], nargs='+', type=int, help="Conv1D kernels")
parser.add_argument("--lstm_layers", default=1, type=int, help="number of LSTM layers pr block")
parser.add_argument("--hidden_size", default=256, type=int, help="size of the LSTM layers")
parser.add_argument("--dropout_prob", default=0.0, type=float, help="size of the LSTM layers")

parser.add_argument("--epochs", default=20, type=int, help="number of epochs")
parser.add_argument("--warm_up", default=10, type=int, help="epochs before lr annealing starts")
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
    project="dae-astro-libri",
    group=None,
)
wandb.config.update(args)

rich.print(vars(args))

token_map_out = TokenMap(tokens=ENGLISH_STANDARD, add_blank=True)
blank_token_idx = token_map_out.token2index[BLANK_TOKEN]
output_size = len(token_map_out)

text_loader = TextLoader("txt", cache=False)
text_cleaner = TextCleaner(lambda s: s.lower().strip())
encode_int_out = EncodeInteger(token_map=token_map_out, tokenizer=char_tokenizer)
text_transform = Compose(
    text_cleaner,
    encode_int_out
    
)
text_batcher = TextBatcher()

token_map_in = TokenMap(tokens=ENGLISH_STANDARD)
encode_int_in = EncodeInteger(
    token_map=token_map_in,
    tokenizer=char_tokenizer
)
astro_speech = AstroSpeech(
    num_tokens=len(token_map_in),
    whitespace_idx=token_map_in.token2index[" "],
    duration=args.duration,
    fade=args.fade,
    mel_delta=args.mel_delta,
    token_shift=args.token_shift
)
audio_transform = Compose(
    text_cleaner,
    encode_int_in,
    astro_speech
)
audio_batcher = AudioBatcher()

modalities = [
    (text_loader, audio_transform, audio_batcher),
    (text_loader, text_transform, text_batcher)
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
    max_len=float(args.sample_rate * args.seconds_pr_batch),
    max_pool_difference=float(args.sample_rate * args.max_second_diff),
    num_batches=2000
)

val_sampler = LengthEvalSampler(
    source=LIBRISPEECH_DEV_CLEAN,
    field="length.flac.samples",
    max_len=float(args.sample_rate * args.seconds_pr_batch)
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


model = AstroDAE(
    token_map=token_map_out,
    in_channels=1,
    kernels=args.conv_kernels,
    hidden_size=args.hidden_size,
    lstm_layers=args.lstm_layers,
    bits=8
)

model.to(device)
model.set_transform_device(device)

optimizer = getattr(torch.optim, args.optimizer)
optimizer = optimizer(model.parameters(), lr=args.lr_max, **args.optimizer_kwargs)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - args.warm_up), eta_min=args.lr_min)

tracker = Tracker()
p = list(model.parameters())
c = 0

for epoch in tracker.epochs(args.epochs):

    # update hyperparams (post)
    tau = 1.0 #4.0 - ((epoch - 1)/(args.epochs - 1)) * 3.9

    # training
    model.train()
    for ((x, x_sl), (y, y_sl)), metadata in tracker.steps(train_loader):

        x = x.to(device)
        y = y.to(device)

        loss, metrics, outputs = model(x, x_sl, y, y_sl, tau=tau)

        optimizer.zero_grad()
        loss.backward()
        
        # if c % 500 == 0:
        #     print("\n" * 2 + "-" * 50)
        #     for p_ in p:
        #         print(p_.shape, p_.grad.abs().sum())
        #     print("-" * 50 + "\n" * 2)
        # c += 1

        optimizer.step()

        tracker.update(metrics)

    tracker.reset()

    # evaluation
    model.eval()
    with torch.no_grad():
        for ((x, x_sl), (y, y_sl)), metadata in tracker.steps(val_loader):
            
            x = x.to(device)
            y = y.to(device)

            loss, metrics, outputs = model(x, x_sl, y, y_sl)

            tracker.update(metrics)

    tracker.reset()

    # update hyperparams (post)
    if epoch >= args.warm_up:
        lr_scheduler.step()
