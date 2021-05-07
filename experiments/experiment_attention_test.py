import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#os.environ["WANDB_MODE"] = "disabled" # equivalent to "wandb disabled"

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
from vseq.data.datapaths import PENN_TREEBANK_TEST, PENN_TREEBANK_TRAIN, PENN_TREEBANK_VALID
from vseq.data.tokens import DELIMITER_TOKEN, ENGLISH_STANDARD, PENN_TREEBANK_ALPHABET, UNKNOWN_TOKEN
from vseq.data.tokenizers import char_tokenizer, word_tokenizer
from vseq.data.loaders import TextLoader
from vseq.data.token_map import TokenMap
from vseq.data.transforms import Compose, EncodeInteger, TextCleaner
from vseq.data.vocabulary import load_vocabulary
from vseq.evaluation import Tracker
from vseq.utils.rand import set_seed, get_random_seed
from vseq.utils.argparsing import str2bool


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--optimizer", default='Adam', type=str, help="optimizer")
parser.add_argument("--optimizer_kwargs", default='{}', type=json.loads, help="extra kwargs for optimizer")
parser.add_argument("--embedding_dim", default=64, type=int, help="dimensionality of embedding space")
parser.add_argument("--hidden_size", default=64, type=int, help="dimensionality of hidden state in LSTM")
parser.add_argument("--num_layers", default=2, type=int, help="number of encoder LSTM layers")
parser.add_argument("--word_dropout", default=0.34, type=float, help="word dropout probability")
parser.add_argument("--token_level", default="word", type=str, choices=["word", "char"], help="word- or character-level modelling")
parser.add_argument("--epochs", default=250, type=int, help="number of epochs")
parser.add_argument("--cache_dataset", default=True, type=str2bool, help="if True, cache the dataset in RAM")
parser.add_argument("--num_workers", default=4, type=int, help="number of dataloader workers")
parser.add_argument("--seed", default=None, type=int, help="random seed")
parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])

args, _ = parser.parse_known_args()


if args.seed is None:
    args.seed = get_random_seed()

set_seed(args.seed)

device = vseq.utils.device.get_device() if args.device == "auto" else torch.device(args.device)


wandb.init(
    entity="vseq",
    project="att-identity",
    group=None,
)
wandb.config.update(args)

rich.print(vars(args))


tokens = PENN_TREEBANK_ALPHABET
token_map = TokenMap(tokens=tokens, add_unknown=True, add_delimit=True)
char_transform = Compose(
    TextCleaner(lambda s: s.replace("<unk>", UNKNOWN_TOKEN)),
    EncodeInteger(token_map=token_map, tokenizer=char_tokenizer)
)

batcher = TextBatcher()
loader = TextLoader('txt', cache=True)

modalities = [(loader, char_transform, batcher),
]

train_dataset = BaseDataset(
    source=PENN_TREEBANK_TRAIN,
    modalities=modalities,
)
val_dataset = BaseDataset(
    source=PENN_TREEBANK_VALID,
    modalities=modalities,
)
# test_dataset = BaseDataset(
#     source=PENN_TREEBANK_TEST,
#     modalities=modalities,
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
# test_loader = DataLoader(
#     dataset=test_dataset,
#     collate_fn=test_dataset.collate,
#     num_workers=args.num_workers,
#     shuffle=False,
#     batch_size=args.batch_size,
# )

delimiter_token_idx = token_map.get_index(DELIMITER_TOKEN)

encoder = vseq.models.LSTMEncoder(
    num_embeddings=len(token_map),
    embedding_dim=args.embedding_dim,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers
)

decoder = vseq.models.BahdanauAttentionDecoder(
    num_embeddings=len(token_map),
    embedding_dim=args.embedding_dim,
    memory_dim=args.hidden_size,
    att_dim=args.hidden_size,
    hidden_size=args.hidden_size,
    num_outputs=len(token_map),
    delimiter_token_idx=delimiter_token_idx
)



#wandb.watch(model, log="all", log_freq=len(train_loader))

encoder = encoder.to(device)
decoder = decoder.to(device)

optimizer = getattr(torch.optim, args.optimizer)
optimizer = optimizer(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=args.lr,
    **args.optimizer_kwargs
)

print(encoder)
print(decoder)

tracker = Tracker()

# for epoch in tracker.epochs(args.epochs):

# model.train()
for (x, x_sl), metadata in tracker.steps(train_loader):
    
    # prepare data
    x = x.to(device)
    y = x[:, 1:].clone().to(device)
    y_sl = x_sl - 1

    memory = encoder(x, x_sl, word_dropout_rate=args.word_dropout)
    decoder.forward(memory, y, y_sl)
    
    # forward pass


    break

    #     loss, metrics, outputs = model(x, x_sl, word_dropout_rate=args.word_dropout, loss_reduction=args.loss_reduction)

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     tracker.update(metrics)

    # model.eval()
    # with torch.no_grad():
    #     for (x, x_sl), metadata in tracker.steps(val_loader):
    #         x = x.to(device)

    #         loss, metrics, outputs = model(x, x_sl, loss_reduction=args.loss_reduction)

    #         tracker.update(metrics)

    # tracker.log()
