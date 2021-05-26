import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#os.environ["WANDB_MODE"] = "disabled" # equivalent to "wandb disabled"

import argparse
import logging
import json

import torch
import wandb
import rich

from torch.utils.data import DataLoader
import torch.nn.functional as F

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
parser.add_argument("--embedding_dim", default=256, type=int, help="dimensionality of embedding space")
parser.add_argument("--hidden_size", default=256, type=int, help="dimensionality of hidden state in LSTM")
parser.add_argument("--num_layers", default=1, type=int, help="number of encoder LSTM layers")
parser.add_argument("--word_dropout", default=0.0, type=float, help="word dropout probability")
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
    project="att-char-to-word",
    group=None,
)
wandb.config.update(args)

rich.print(vars(args))


char_tokens = PENN_TREEBANK_ALPHABET
char_token_map = TokenMap(tokens=char_tokens, add_unknown=True, add_delimit=False)

word_tokens = load_vocabulary(PENN_TREEBANK_TRAIN)
word_tokens.remove("<unk>")
word_token_map = TokenMap(tokens=word_tokens, add_unknown=True, add_delimit=True)

word_set = set(word_tokens)
def word_replace_unk(s):
    return " ".join([w if w in word_set else UNKNOWN_TOKEN for w in s.split()])


char_transform = Compose(
    TextCleaner(lambda s: word_replace_unk(s.replace("<unk>", UNKNOWN_TOKEN).strip())),
    EncodeInteger(token_map=char_token_map, tokenizer=char_tokenizer)
)

word_transform = Compose(
    TextCleaner(lambda s: s.replace("<unk>", UNKNOWN_TOKEN).strip()),
    EncodeInteger(token_map=word_token_map, tokenizer=word_tokenizer)
)

batcher = TextBatcher()
loader = TextLoader('txt', cache=True)

modalities = [
    (loader, char_transform, batcher),
    (loader, word_transform, batcher)
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

delimiter_token_idx = char_token_map.get_index(DELIMITER_TOKEN)

encoder = vseq.models.LSTMEncoder(
    num_embeddings=len(char_token_map),
    embedding_dim=args.embedding_dim,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers
)

decoder = vseq.models.MultiheadAttentionDecoder(
    num_embeddings=len(word_token_map),
    hidden_size=args.hidden_size,
    num_heads=4,
    num_outputs=len(word_token_map),
    delimiter_token_idx=delimiter_token_idx
)

reconstruct = vseq.models.OPReconstruct(
    num_embeddings=len(word_token_map),
    embedding_dim=args.embedding_dim,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    num_outputs=len(char_token_map),
    embed_before_outer_product=True
)

# wandb.watch(model, log="all", log_freq=len(train_loader))

encoder = encoder.to(device)
decoder = decoder.to(device)
reconstruct = reconstruct.to(device)

optimizer = getattr(torch.optim, args.optimizer)
optimizer = optimizer(
    list(encoder.parameters()) + list(decoder.parameters()) + list(reconstruct.parameters()),
    lr=args.lr,
    **args.optimizer_kwargs
)

print(encoder)
print(decoder)

tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    encoder.train()
    decoder.train()
    reconstruct.train()
    for ((x, x_sl), (y, y_sl)), metadata in tracker.steps(train_loader):
        
        # prepare data
        x = x.to(device)
        y = y.to(device)
        memory_sl = x_sl
        aw_sl = y_sl - 2 # - 2 because no start is predicted and we dont care about the end

        memory = encoder(x, x_sl, word_dropout_rate=args.word_dropout)
        loss_inf, logits, log_prob, metrics_inf, p_y, aw = decoder.forward(memory, memory_sl, y, y_sl)
        loss_rec, metrics_rec, p_x = reconstruct(logits, aw, aw_sl, x, x_sl)
        loss = loss_inf + loss_rec

        tracker.update(metrics_inf + metrics_rec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tracker.reset()

    encoder.eval()
    decoder.eval()
    reconstruct.eval()
    for ((x, x_sl), (y, y_sl)), metadata in tracker.steps(val_loader):
        
        # prepare data
        x = x.to(device)
        y = y.to(device)
        memory_sl = x_sl
        aw_sl = y_sl - 2 # - 2 because no start is predicted and we dont care about the end

        memory = encoder(x, x_sl, word_dropout_rate=args.word_dropout)
        loss_inf, logits, log_prob, metrics_inf, p_y, aw = decoder.forward(memory, memory_sl, y, y_sl)
        loss_rec, metrics_rec, p_x = reconstruct(logits, aw, aw_sl, x, x_sl)
        loss = loss_inf + loss_rec

        tracker.update(metrics_inf + metrics_rec)

    tracker.reset()
    

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
