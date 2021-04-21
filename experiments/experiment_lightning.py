import argparse
import logging

import torch
import rich
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.loggers.wandb import WandbLogger

import vseq.models

from vseq.data import BaseDataset
from vseq.data.batchers import TextBatcher
from vseq.data.datapaths import PENN_TREEBANK_TEST, PENN_TREEBANK_TRAIN, PENN_TREEBANK_VALID
from vseq.data.tokens import DELIMITER_TOKEN
from vseq.data.tokenizers import word_tokenizer
from vseq.data.token_map import TokenMap
from vseq.data.transforms import EncodeInteger
from vseq.data.vocabulary import load_vocabulary
from vseq.models import PLModelWrapper
from vseq.utils.rand import  get_random_seed
from vseq.utils.argparsing import str2bool


LOGGER = logging.getLogger(name=__file__)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--embedding_dim", default=353, type=int, help="dimensionality of embedding space")
parser.add_argument("--hidden_size", default=191, type=int, help="dimensionality of hidden state in LSTM")
parser.add_argument("--word_dropout", default=0.38, type=float, help="word dropout probability")
parser.add_argument(
    "--loss_reduction",
    default="nats_per_dim",
    type=str,
    choices=["nats_per_dim", "nats_per_example"],
    help="loss reduction",
)
parser.add_argument("--cache_dataset", default=True, type=str2bool, help="if True, cache the dataset in RAM")
parser.add_argument("--num_workers", default=4, type=int, help="number of dataloader workers")
parser.add_argument("--seed", default=None, type=int, help="random seed")

parser = pl.Trainer.add_argparse_args(parser)

args, _ = parser.parse_known_args()


if args.seed is None:
    args.seed = get_random_seed()

pl.seed_everything(args.seed)


logger = WandbLogger(
    entity="vseq",
    project="lstmlm",
    group=None,
)
rich.print(vars(args))


vocab = load_vocabulary(PENN_TREEBANK_TRAIN)
token_map = TokenMap(tokens=vocab, add_start=False, add_end=False, add_delimit=True)
penn_treebank_transform = EncodeInteger(
    token_map=token_map,
    tokenizer=word_tokenizer,
)
batcher = TextBatcher()

modalities = [("txt", penn_treebank_transform, batcher)]

train_dataset = BaseDataset(
    source=PENN_TREEBANK_TRAIN,
    modalities=modalities,
    cache=args.cache_dataset,
)
val_dataset = BaseDataset(
    source=PENN_TREEBANK_VALID,
    modalities=modalities,
    cache=args.cache_dataset,
)
test_dataset = BaseDataset(
    source=PENN_TREEBANK_TEST,
    modalities=modalities,
    cache=args.cache_dataset,
)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    shuffle=True,
    batch_size=args.batch_size,
    pin_memory=True
)
val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size,
    pin_memory=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    collate_fn=test_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size,
    pin_memory=True
)


delimiter_token_idx = token_map.get_index(DELIMITER_TOKEN)
model = vseq.models.LSTMLM(
    num_embeddings=len(token_map),
    embedding_dim=args.embedding_dim,
    hidden_size=args.hidden_size,
    delimiter_token_idx=delimiter_token_idx,
)
logger.watch(model, log='all', log_freq=len(train_loader))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


lightning_module = PLModelWrapper(model, optimizer)


if __name__ == '__main__':
    args.auto_select_gpus = True
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(lightning_module, train_loader, val_loader)

    result = trainer.test(test_dataloaders=test_loader)
    print(result)
