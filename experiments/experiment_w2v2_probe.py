import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
#os.environ["WANDB_MODE"] = "disabled" # equivalent to "wandb disabled"

import argparse
import logging
import json
import csv

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
from vseq.data.batchers import AlignmentBatcher, AudioBatcher
from vseq.data.datapaths import LIBRISPEECH_TRAIN, LIBRISPEECH_DEV_CLEAN
from vseq.data.tokens import LIBRI_PHONESET_INFER
from vseq.data.tokenizers import char_tokenizer
from vseq.data.loaders import AlignmentLoader, AudioLoader
from vseq.data.token_map import TokenMap
from vseq.data.transforms import EncodeIntegerAlignment
from vseq.data.samplers import LengthTrainSampler, LengthEvalSampler, SubsetSampler
from vseq.evaluation import Tracker
from vseq.training import set_dropout
from vseq.utils.rand import set_seed, get_random_seed
from vseq.models import MultiProbe

import fairseq

parser = argparse.ArgumentParser()
parser.add_argument("--sample_rate", default=16000, type=int, help="sample rate")
parser.add_argument("--batch_size", default=1, type=int, help="number of mel filterbanks")

parser.add_argument("--lr_max", default=3e-4, type=float, help="start learning rate")
parser.add_argument("--lr_min", default=5e-5, type=float, help="end learning rate")
parser.add_argument("--optimizer", default='Adam', type=str, help="optimizer")
parser.add_argument("--optimizer_kwargs", default='{}', type=json.loads, help="extra kwargs for optimizer")

parser.add_argument("--w2v2_model_path", default='/data/research/users/labo/wav2vec_vox_new.pt', type=str, help="pre-trained model")

parser.add_argument("--epochs", default=10, type=int, help="number of epochs")
parser.add_argument("--warm_up", default=5, type=int, help="number of epochs before starting annealing")
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
    project="asr-ctc-libri",
    group=None,
)
wandb.config.update(args)
rich.print(vars(args))

token_map = TokenMap(tokens=LIBRI_PHONESET_INFER)
output_size = len(token_map)

align_loader = AlignmentLoader(index=1) # index 0 == words, index 1 == phones
align_transform = EncodeIntegerAlignment(token_map=token_map, strip_accent=True)
align_batcher = AlignmentBatcher()

audio_loader = AudioLoader("flac", cache=False, sum_channels=True)
audio_transform = None
audio_batcher = AudioBatcher()

modalities = [
    (audio_loader, audio_transform, audio_batcher),
    (align_loader, align_transform, align_batcher)
]

def filter_examples(dataset, max_secs=24.0):
    """
    Filter away examples without alignments and
    those too long to be processed by W2V 2.0.
    """
    
    with open(dataset.source_filepath, newline='') as source_file_buffer:
        reader = csv.DictReader(source_file_buffer)
        lengths = [int(row["length.flac.samples"]) for row in reader]
    
    to_be_removed = []
    no_tg, too_long = 0, 0
    for idx, example in enumerate(dataset.examples):
        if not os.path.exists(f"{example}.TextGrid"):
            to_be_removed.append(example)
            no_tg += 1
        elif lengths[idx] / args.sample_rate > max_secs:
            to_be_removed.append(example)
            too_long += 1
    
    for example in to_be_removed:
        dataset.examples.remove(example)

    print(f"Removed {len(to_be_removed)} files from {dataset.source}.")


train_dataset = BaseDataset(
    source=LIBRISPEECH_TRAIN,
    modalities=modalities,
)
val_dataset = BaseDataset(
    source=LIBRISPEECH_DEV_CLEAN,
    modalities=modalities,
)

filter_examples(train_dataset)
filter_examples(val_dataset)

train_sampler = SubsetSampler(num_examples=1000, total_examples=len(train_dataset))

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    batch_size=args.batch_size,
    sampler=train_sampler
)

val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    batch_size=args.batch_size
)

w2v2, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.w2v2_model_path])
w2v2 = w2v2[0]
w2v2.to(device)
w2v2.eval()

model = MultiProbe(
    num_probes=len(w2v2.encoder.layers),
    input_size=w2v2.encoder.embedding_dim,
    output_size=output_size
)
model.to(device)


optimizer = getattr(torch.optim, args.optimizer)
optimizer = optimizer(model.parameters(), lr=args.lr_max, **args.optimizer_kwargs)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=(args.epochs - args.warm_up), eta_min=args.lr_min)

tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    model.train()
    for ((x, x_sl), ((y, m, a), y_sl)), metadata in tracker.steps(train_loader):
        x = x.to(device)
        y = y[0].to(device)
        m = m[0].to(device)
        a = a[0]

        with torch.no_grad():
            result = w2v2.forward(x, padding_mask=None, mask=False, features_only=True, layer=100)
            features = [l[0][:, 0][m] for l in result["layer_results"]]
            del result

        loss, outputs, metrics = model(features, y)
        tracker.update(metrics)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    for ((x, x_sl), ((y, m, a), y_sl)), metadata in tracker.steps(val_loader):

        x = x.to(device)
        y = y[0].to(device)
        m = m[0].to(device)
        a = a[0]

        with torch.no_grad():
            result = w2v2.forward(x, padding_mask=None, mask=False, features_only=True, layer=100)
            features = [l[0][:, 0][m] for l in result["layer_results"]]
            del result

            loss, outputs, metrics = model(features, y)
            tracker.update(metrics)
        
    tracker.reset()

    # update hyperparams (post)
    if epoch >= args.warm_up:
        lr_scheduler.step()
