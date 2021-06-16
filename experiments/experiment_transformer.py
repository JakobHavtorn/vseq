import argparse

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
parser.add_argument("--embedding_dim", default=200, type=int, help="dimensionality of embedding space")
parser.add_argument("--hidden_size", default=200, type=int, help="dimension of the feedforward network model in nn.TransformerEncoder")
parser.add_argument("--num_layers", default=2, type=int, help="number of nn.TransformerEncoderLayer in nn.TransformerEncoder")
parser.add_argument("--num_heads", default=2, type=int, help="number of heads in the multiheadattention models")
parser.add_argument("--dropout", default=0.2, type=float, help="inter LSTM layer dropout probability")
parser.add_argument("--token_level", default="word", type=str, choices=["word", "char"], help="word- or character-level modelling")
parser.add_argument("--epochs", default=250, type=int, help="number of epochs")
parser.add_argument("--num_workers", default=4, type=int, help="number of dataloader workers")
parser.add_argument("--seed", default=None, type=int, help="random seed")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args = parser.parse_args()


if args.seed is None:
    args.seed = get_random_seed()

set_seed(args.seed)

device = vseq.utils.device.get_device() if args.device == "auto" else torch.device(args.device)


wandb.init(
    entity="vseq",
    project="transformer",
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


model = vseq.models.TransformerLM(
    num_embeddings=len(token_map),
    embedding_dim=args.embedding_dim,
    hidden_size=args.hidden_size,
    num_heads=args.num_heads,
    num_layers=args.num_layers,
    dropout=args.dropout,
)

model = model.to(device)
print(model)
x, x_sl = next(iter(train_loader))[0]
x = x.to(device)
model.summary(input_data=x[:, :2], x_sl=torch.tensor([2] * x.size(0), dtype=int))
wandb.watch(model, log="all", log_freq=len(train_loader))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker.steps(train_loader):
        x = x.to(device)

        loss, metrics, outputs = model(x, x_sl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)

    model.eval()
    with torch.no_grad():
        for (x, x_sl), metadata in tracker.steps(val_loader):
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl)

            tracker.update(metrics)

        for (x, x_sl), metadata in tracker.steps(test_loader):
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl)

            tracker.update(metrics)

    tracker.log()
