import os

from numpy.lib.financial import ipmt
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["WANDB_MODE"] = "disabled" # equivalent to "wandb disabled"

import argparse
import logging
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import rich
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import vseq
import vseq.data
import vseq.models
import vseq.utils
import vseq.utils.device

from vseq.data import BaseDataset
from vseq.data.batchers import TextBatcher, AudioBatcher
from vseq.data.datapaths import LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_TRAIN
from vseq.data.tokens import ENGLISH_STANDARD, UNKNOWN_TOKEN
from vseq.data.tokenizers import char_tokenizer
from vseq.data.loaders import TextLoader
from vseq.data.token_map import TokenMap
from vseq.data.transforms import Compose, EncodeInteger, TextCleaner, AstroSpeech
from vseq.data.samplers import LengthTrainSampler, LengthEvalSampler
from vseq.evaluation import Tracker
from vseq.utils.rand import set_seed, get_random_seed
from vseq.models import AstroVAE
from vseq.training.saving import save_exp_file, save_model

parser = argparse.ArgumentParser()

parser.add_argument("--sample_rate", default=800, type=int, help="sample rate")
parser.add_argument("--duration", default=50, type=int, help="number of frames for each astro speech token")
parser.add_argument("--fade", default=15, type=int, help="cos-annealing duration for fading in/out astro tokens")
# parser.add_argument("--min_mel", default=200, type=int, help="the mel-frequency shift between tokens")
parser.add_argument("--mel_delta", default=10, type=int, help="the mel-frequency shift between tokens")
parser.add_argument("--token_shift", default=0, type=int, help="frequency span for each astro token")

parser.add_argument("--max_len", default=5000, type=int, help="max sum of batch lengths")
parser.add_argument("--max_pool_difference", default=50, type=int, help="max char variation in batch")

parser.add_argument("--lr_max", default=3e-4, type=float, help="start learning rate")
parser.add_argument("--lr_min", default=5e-5, type=float, help="end learning rate")
parser.add_argument("--optimizer", default='Adam', type=str, help="optimizer")
parser.add_argument("--optimizer_kwargs", default='{}', type=json.loads, help="extra kwargs for optimizer")

parser.add_argument("--kernel_and_strides", default=[10, 5], type=list, help="kernel and stride size of encoder and decoder")
parser.add_argument("--conv_hidden_size", default=256, type=int, help="innner dimension of the text-VAE")
parser.add_argument("--hidden_size", default=128, type=int, help="innner dimension of the text-VAE")
parser.add_argument("--prior_hidden_size", default=512, type=int, help="innner dimension of the prior")
parser.add_argument("--output_bits", default=8, type=int, help="bits used for the output distribution")

parser.add_argument("--epochs", default=1000, type=int, help="number of epochs")
parser.add_argument("--warm_up", default=500, type=int, help="epochs before lr annealing starts")
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
    project="vae-text",
    group=None,
)
wandb.config.update(args)
tracking_enabled = not (wandb.run.dir == "/")

if tracking_enabled:
    save_exp_file(wandb.run.dir)

rich.print(vars(args))

token_map = TokenMap(tokens=ENGLISH_STANDARD, add_unknown=True)
output_size = len(token_map)

text_loader = TextLoader("txt", cache=True)
text_cleaner = TextCleaner(lambda s: s.lower().strip())
encode_int_out = EncodeInteger(token_map=token_map, tokenizer=char_tokenizer)
text_transform = Compose(
    text_cleaner,
    encode_int_out
)
text_batcher = TextBatcher()

encode_int_in = EncodeInteger(
    token_map=token_map,
    tokenizer=char_tokenizer
)
astro_speech = AstroSpeech(
    num_tokens=len(token_map),
    whitespace_idx=token_map.token2index[" "],
    sample_rate=args.sample_rate,
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
    field="length.txt.chars",
    max_len=float(args.max_len),
    max_pool_difference=float(args.max_pool_difference),
    num_batches=1000
)

val_sampler = LengthEvalSampler(
    source=LIBRISPEECH_DEV_CLEAN,
    field="length.txt.chars",
    max_len=float(args.max_len),
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

model = AstroVAE(
    kernel_and_strides=args.kernel_and_strides,
    hidden_size=args.hidden_size,
    conv_hidden_size=args.conv_hidden_size,
    prior_hidden_size=args.prior_hidden_size,
    num_components=len(token_map),
    output_bits=args.output_bits
)
model.to(device)
model.set_transform_device(device)

# load prior
# path = "/home/labo/repos/vseq/experiments/astro/text_vae/wandb/run-20210722_131840-1mwokzpw/files/prior.pt"
# prior_state_dict = torch.load(path)
# model.load_state_dict(prior_state_dict, strict=False)
# parameters = [v for k, v in model.state_dict().items() if k not in prior_state_dict]
# assert len(parameters) + len(prior_state_dict) == len(model.state_dict())

optimizer = getattr(torch.optim, args.optimizer)
optimizer = optimizer(model.parameters(), lr=args.lr_max, **args.optimizer_kwargs)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - args.warm_up), eta_min=args.lr_min)

tracker = Tracker()

get_best_loss = lambda: tracker.best_metrics[LIBRISPEECH_DEV_CLEAN]["best_loss"].value
current_best = float("inf")

num_subset_examples = len(train_dataset)
num_processed = 0

for epoch in tracker.epochs(args.epochs):

    beta = 1.0 # min((epoch - 1) / 3, 1.0)
    
    # training
    model.train()
    for ((x, x_sl), (y, y_sl)), metadata in tracker.steps(train_loader):

        x = x.to(device)

        loss, metrics, outputs = model(x, x_sl, beta=beta)

        #import IPython; IPython.embed()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)

    # evaluation
    model.eval()
    with torch.no_grad():
        for ((x, x_sl), (y, y_sl)), metadata in tracker.steps(val_loader):
            
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl)

            tracker.update(metrics)
    
    # # generate samples
    # samples = model.generate(num_examples=10, max_len=100, device=device)
    # samples_sl = torch.full([10], 100)
    # samples = token_map.decode_batch(samples, samples_sl, join_separator="")
    # data = [(i, t) for i, t in enumerate(samples)]
    # text_table = wandb.Table(columns=["idx", "samples"], data=data)
    
    tracker.log()

    # update hyperparams (post)
    if epoch >= args.warm_up:
        lr_scheduler.step()
    
    # save model
    if tracking_enabled and get_best_loss() < current_best:
        save_model(wandb.run.dir, model.state_dict(), "model")
        #save_model(wandb.run.dir, model.state_dict_prior(), "prior")
        current_best = get_best_loss()
        print("Model saved...\n")
        