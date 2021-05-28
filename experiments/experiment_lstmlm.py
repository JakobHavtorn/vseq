import argparse
import json

import torch
import wandb
import rich

from torch.utils.data import DataLoader

import vseq
import vseq.data
import vseq.models
import vseq.utils
import vseq.utils.device

from vseq.data import BaseDataset
from vseq.data.batchers import TextBatcher
from vseq.data.datapaths import PENN_TREEBANK_TEST, PENN_TREEBANK_TRAIN, PENN_TREEBANK_VALID
from vseq.data.loaders import TextLoader
from vseq.data.tokens import DELIMITER_TOKEN, ENGLISH_STANDARD, PENN_TREEBANK_ALPHABET, UNKNOWN_TOKEN
from vseq.data.tokenizers import char_tokenizer, word_tokenizer
from vseq.data.token_map import TokenMap
from vseq.data.transforms import Compose, EncodeInteger, TextCleaner
from vseq.data.vocabulary import load_vocabulary
from vseq.evaluation import Tracker
from vseq.utils.rand import set_seed, get_random_seed
from vseq.utils.argparsing import str2bool


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--optimizer_json", default='{"optimizer": "Adam"}', type=json.loads, help="extra kwargs for optimizer")
parser.add_argument("--embedding_dim", default=464, type=int, help="dimensionality of embedding space")
parser.add_argument("--hidden_size", default=373, type=int, help="dimensionality of hidden state in LSTM")
parser.add_argument("--num_layers", default=1, type=int, help="number of LSTM layers")
parser.add_argument("--dropout", default=0.0, type=float, help="inter LSTM layer dropout probability")
parser.add_argument("--word_dropout", default=0.34, type=float, help="word dropout probability")
parser.add_argument("--layer_norm", default=False, type=str2bool, help="use layer normalization")
parser.add_argument("--token_level", default="word", type=str, choices=["word", "char"], help="word- or character-level modelling")
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

if args.token_level == "word":
    tokens = load_vocabulary(PENN_TREEBANK_TRAIN)
    token_map = TokenMap(tokens=tokens, add_delimit=True)
    penn_treebank_transform = EncodeInteger(token_map=token_map, tokenizer=word_tokenizer)
else:
    tokens = PENN_TREEBANK_ALPHABET
    token_map = TokenMap(tokens=tokens, add_delimit=True, add_unknown=True)
    penn_treebank_transform = Compose(
        TextCleaner(lambda s: s.replace("<unk>", UNKNOWN_TOKEN)),
        EncodeInteger(token_map=token_map, tokenizer=char_tokenizer)
    )

batcher = TextBatcher()
loader = TextLoader('txt', cache=True)

modalities = [(loader, penn_treebank_transform, batcher)]

train_dataset = BaseDataset(
    source=PENN_TREEBANK_TRAIN,
    modalities=modalities,
)
val_dataset = BaseDataset(
    source=PENN_TREEBANK_VALID,
    modalities=modalities,
)
test_dataset = BaseDataset(
    source=PENN_TREEBANK_TEST,
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
test_loader = DataLoader(
    dataset=test_dataset,
    collate_fn=test_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size,
    pin_memory=True,
)


delimiter_token_idx = token_map.get_index(DELIMITER_TOKEN)
model = vseq.models.LSTMLM(
    num_embeddings=len(token_map),
    embedding_dim=args.embedding_dim,
    hidden_size=args.hidden_size,
    layer_norm=args.layer_norm,
    dropout=args.dropout,
    delimiter_token_idx=delimiter_token_idx,
)

# wandb.watch(model, log="all", log_freq=len(train_loader))
model = model.to(device)
print(model)
# x, x_sl = next(iter(train_loader))[0]
# x = x.to(device)
# model.summary(input_data=x, x_sl=x_sl)

optimizer = args.optimizer_json.pop('optimizer')
optimizer = getattr(torch.optim, optimizer)
optimizer = optimizer(model.parameters(), lr=args.lr, **args.optimizer_json)

tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker(train_loader):
        x = x.to(device)

        loss, metrics, outputs = model(x, x_sl, word_dropout_rate=args.word_dropout)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)

    model.eval()
    with torch.no_grad():
        for (x, x_sl), metadata in tracker(val_loader):
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl)

            tracker.update(metrics)

        for (x, x_sl), metadata in tracker(test_loader):
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl)

            tracker.update(metrics)


    # Log tracker metrics
    tracker.log()
