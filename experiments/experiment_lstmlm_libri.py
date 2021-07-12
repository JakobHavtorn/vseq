import argparse
import logging
import json

import torch
import wandb
import rich

from torch.utils.data import DataLoader

import vseq
import vseq.data
from vseq.data import tokenizers
import vseq.models
import vseq.utils
import vseq.utils.device

from vseq.data import BaseDataset
from vseq.data.batchers import TextBatcher
from vseq.data.datapaths import LIBRISPEECH_TRAIN, LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_TEST_CLEAN
from vseq.data.tokens import DELIMITER_TOKEN, ENGLISH_STANDARD, PENN_TREEBANK_ALPHABET, UNKNOWN_TOKEN
from vseq.data.tokenizers import char_tokenizer, word_tokenizer
from vseq.data.loaders import TextLoader
from vseq.data.token_map import TokenMap
from vseq.data.transforms import Compose, EncodeInteger, TextCleaner
from vseq.data.vocabulary import load_vocabulary
from vseq.evaluation import Tracker
from vseq.utils.rand import set_seed, get_random_seed
from vseq.utils.argparsing import str2bool
from vseq.data.samplers import LengthTrainSampler, LengthEvalSampler


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--optimizer", default='Adam', type=str, help="optimizer")
parser.add_argument("--optimizer_kwargs", default='{}', type=json.loads, help="extra kwargs for optimizer")
parser.add_argument("--embedding_dim", default=464, type=int, help="dimensionality of embedding space")
parser.add_argument("--hidden_size", default=373, type=int, help="dimensionality of hidden state in LSTM")
parser.add_argument("--word_dropout", default=0.34, type=float, help="word dropout probability")
parser.add_argument(
    "--loss_reduction",
    default="nats_per_dim",
    type=str,
    choices=["nats_per_dim", "nats_per_example"],
    help="loss reduction",
)
parser.add_argument("--epochs", default=250, type=int, help="number of epochs")
parser.add_argument("--cache_dataset", default=True, type=str2bool, help="if True, cache the dataset in RAM")
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
    project="lstmlm",
    group=None,
)
wandb.config.update(args)
rich.print(vars(args))


tokens = ENGLISH_STANDARD
token_map = TokenMap(tokens=tokens, add_delimit=True, add_unknown=True)
penn_treebank_transform = Compose(
    TextCleaner(lambda s: s.lower().strip()),
    EncodeInteger(token_map=token_map, tokenizer=char_tokenizer)
)

batcher = TextBatcher()
loader = TextLoader('txt', cache=True)

modalities = [(loader, penn_treebank_transform, batcher)]

train_dataset = BaseDataset(
    source=LIBRISPEECH_TRAIN,
    modalities=modalities,
)
val_dataset = BaseDataset(
    source=LIBRISPEECH_DEV_CLEAN,
    modalities=modalities,
)
test_dataset = BaseDataset(
    source=LIBRISPEECH_TEST_CLEAN,
    modalities=modalities,
)

train_sampler = LengthTrainSampler(
    source=LIBRISPEECH_TRAIN,
    field="length.txt.chars",
    max_len=float(args.max_len),
    max_pool_difference=float(args.max_pool_difference),
    num_batches=2000
)

# val_sampler = LengthEvalSampler(
#     source=LIBRISPEECH_DEV_CLEAN,
#     field="length.txt.chars",
#     max_len=float(args.max_len),
# )

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    shuffle=True,
    batch_size=args.batch_size,
)
val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size,
)
test_loader = DataLoader(
    dataset=test_dataset,
    collate_fn=test_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size,
)

delimiter_token_idx = token_map.get_index(DELIMITER_TOKEN)
model = vseq.models.LSTMLM(
    num_embeddings=len(token_map),
    embedding_dim=args.embedding_dim,
    hidden_size=args.hidden_size,
    delimiter_token_idx=delimiter_token_idx,
)

wandb.watch(model, log="all", log_freq=len(train_loader))

model = model.to(device)

optimizer = getattr(torch.optim, args.optimizer)
optimizer = optimizer(model.parameters(), lr=args.lr, **args.optimizer_kwargs)
print(model)

x, x_sl = next(iter(train_loader))[0]
x = x.to(device)
# print(model.summary(input_example=x, x_sl=x_sl))


tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker(train_loader):
        x = x.to(device)

        loss, metrics, outputs = model(x, x_sl, word_dropout_rate=args.word_dropout, loss_reduction=args.loss_reduction)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)

    model.eval()
    with torch.no_grad():
        for (x, x_sl), metadata in tracker(val_loader):
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl, loss_reduction=args.loss_reduction)

            tracker.update(metrics)

        for (x, x_sl), metadata in tracker(test_loader):
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl, loss_reduction=args.loss_reduction)

            tracker.update(metrics)


    # Log tracker metrics
    tracker.log()
