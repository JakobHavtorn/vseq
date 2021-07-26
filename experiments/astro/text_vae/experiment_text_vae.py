import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from vseq.data.batchers import TextBatcher
from vseq.data.datapaths import LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_LM
from vseq.data.tokens import ENGLISH_STANDARD, UNKNOWN_TOKEN
from vseq.data.tokenizers import char_tokenizer
from vseq.data.loaders import TextLoader
from vseq.data.token_map import TokenMap
from vseq.data.transforms import Compose, EncodeInteger, TextCleaner
from vseq.data.samplers import LengthTrainSampler, LengthEvalSampler
from vseq.evaluation import Tracker
from vseq.utils.rand import set_seed, get_random_seed
from vseq.models import TextVAE
from vseq.training.saving import save_exp_file, save_model

parser = argparse.ArgumentParser()

parser.add_argument("--max_len", default=15000, type=int, help="max sum of batch lengths")
parser.add_argument("--max_pool_difference", default=50, type=int, help="max char variation in batch")

parser.add_argument("--lr_max", default=3e-4, type=float, help="start learning rate")
parser.add_argument("--lr_min", default=5e-5, type=float, help="end learning rate")
parser.add_argument("--optimizer", default='Adam', type=str, help="optimizer")
parser.add_argument("--optimizer_kwargs", default='{}', type=json.loads, help="extra kwargs for optimizer")

parser.add_argument("--hidden_size", default=128, type=int, help="innner dimension of the text-VAE")
parser.add_argument("--prior_hidden_size", default=512, type=int, help="innner dimension of the prior")

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
save_exp_file(wandb.run.dir)

rich.print(vars(args))

token_map = TokenMap(tokens=ENGLISH_STANDARD, add_unknown=True)
output_size = len(token_map)

text_val_loader = TextLoader("txt", cache=True)
text_cleaner = TextCleaner(lambda s: s.lower().replace("<unk>", UNKNOWN_TOKEN).strip())
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

val_modalities = [
    (text_val_loader, text_transform, text_batcher)
]

def prepare_train_data(idx):
    
    libri_lm_subset = f"/mnt/data/research/source/librispeech_lm/librispeech_lm_{idx}.txt"
    
    
    text_train_loader = TextLoader("txt", cache=True)
    
    train_modalities = [
    (text_train_loader, text_transform, text_batcher)
    ]

    train_dataset = BaseDataset(
        source=libri_lm_subset,
        modalities=train_modalities,
    )
    train_dataset.source = "librispeech_lm"

    train_sampler = LengthTrainSampler(
        source=train_dataset,
        field=0,
        max_len=float(args.max_len),
        max_pool_difference=float(args.max_pool_difference),
        num_batches=1000
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=train_dataset.collate,
        num_workers=args.num_workers,
        batch_sampler=train_sampler
    )
    
    return train_dataset, train_sampler, train_loader

current_subset_idx = 0
train_dataset, train_sampler, train_loader = prepare_train_data(current_subset_idx)

val_dataset = BaseDataset(
    source=LIBRISPEECH_DEV_CLEAN,
    modalities=val_modalities,
)

val_sampler = LengthEvalSampler(
    source=LIBRISPEECH_DEV_CLEAN,
    field="length.txt.chars",
    max_len=float(args.max_len),
)

val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    batch_sampler=val_sampler
)

model = TextVAE(
    hidden_size=args.hidden_size,
    prior_hidden_size=args.prior_hidden_size,
    token_map=token_map
)
model.to(device)

optimizer = getattr(torch.optim, args.optimizer)
optimizer = optimizer(model.parameters(), lr=args.lr_max, **args.optimizer_kwargs)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - args.warm_up), eta_min=args.lr_min)

tracker = Tracker()

get_best_loss = lambda: tracker.best_metrics[LIBRISPEECH_DEV_CLEAN]["best_loss"].value
current_best = float("inf")

num_subset_examples = len(train_dataset)
num_processed = 0

for epoch in tracker.epochs(args.epochs):

    beta = 1.0 #min(epoch / args.warm_up, 1.0)
    
    # training
    model.train()
    for (x, x_sl), metadata in tracker.steps(train_loader):

        x = x.to(device)
        num_processed += x.size(0)

        loss, metrics, outputs = model(x, x_sl, beta=beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)

    # evaluation
    model.eval()
    with torch.no_grad():
        for (x, x_sl), metadata in tracker.steps(val_loader):
            
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl)

            tracker.update(metrics)
    
    samples = model.generate(num_examples=10, max_len=100, device=device)
    samples_sl = torch.full([10], 100)
    samples = token_map.decode_batch(samples, samples_sl, join_separator="")
    data = [(i, t) for i, t in enumerate(samples)]
    text_table = wandb.Table(columns=["idx", "samples"], data=data)
    
    tracker.log(samples=text_table)

    # update hyperparams (post)
    if epoch >= args.warm_up:
        lr_scheduler.step()
        
    # update examples if done with the subset
    pct_done = num_processed / num_subset_examples
    if pct_done >= 1.0:
        current_subset_idx = (current_subset_idx + 1) % 40
        train_dataset, train_sampler, train_loader = prepare_train_data(current_subset_idx)
        num_subset_examples = len(train_dataset)
        num_processed = 0
        print(f"\nSubset {current_subset_idx} loaded...\n")
    else:
        print(f"\nSubset {current_subset_idx}: {pct_done * 100:.2f}%\n")
    
    # save model
    if get_best_loss() < current_best:
        save_model(wandb.run.dir, model.state_dict(), "model")
        save_model(wandb.run.dir, model.state_dict_prior(), "prior")
        current_best = get_best_loss()
        print("Model saved...\n")
        