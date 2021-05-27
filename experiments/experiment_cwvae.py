import argparse
import logging

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
from vseq.data.tokens import DELIMITER_TOKEN, PENN_TREEBANK_ALPHABET, UNKNOWN_TOKEN
from vseq.data.tokenizers import char_tokenizer, word_tokenizer
from vseq.data.token_map import TokenMap
from vseq.data.transforms import EncodeInteger, TextCleaner, Compose
from vseq.data.vocabulary import load_vocabulary
from vseq.evaluation import Tracker
from vseq.utils.argparsing import str2bool
from vseq.utils.rand import set_seed, get_random_seed
from vseq.training import CosineAnnealer


LOGGER = logging.getLogger(name=__file__)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--embedding_dim", default=256, type=int, help="dimensionality of character/word embedding")
parser.add_argument("--hidden_size", default=512, type=int, help="dimensionality of deterministic variable")
parser.add_argument("--latent_dims", default=[16, 32, 64], type=int, nargs="+", help="dimensionality of latent variables")
parser.add_argument("--clock_rates", default=[1, 2, 4], type=int, nargs="+", help="clockrates of latent spaces")
parser.add_argument("--anneal_steps", default=20000, type=int, help="number of steps to anneal beta")
parser.add_argument("--anneal_start_value", default=0, type=float, help="initial beta annealing value")
parser.add_argument("--prior_samples", default=32, type=int, help="number of prior samples for logging")
parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
parser.add_argument("--cache_dataset", default=True, type=str2bool, help="if True, cache the dataset in RAM")
parser.add_argument("--num_workers", default=8, type=int, help="number of dataloader workers")
parser.add_argument("--wandb_group", default=None, type=str, help='custom group for this experiment (optional)')
parser.add_argument("--seed", default=None, type=int, help="seed for random number generators. Random if -1.")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args, _ = parser.parse_known_args()


if args.seed is None:
    args.seed = get_random_seed()

set_seed(args.seed)

device = vseq.utils.device.get_device() if args.device == "auto" else torch.device(args.device)


wandb.init(
    entity="vseq",
    project="cw-vae",
    group=None,
)
wandb.config.update(args)
rich.print(vars(args))


token_map = TokenMap(tokens=PENN_TREEBANK_ALPHABET, add_delimit=True, add_unknown=True)
penn_treebank_transform = Compose(
    TextCleaner(lambda s: s.replace("<unk>", UNKNOWN_TOKEN)),
    EncodeInteger(token_map=token_map, tokenizer=char_tokenizer)
)

batcher = TextBatcher()
loader = TextLoader('txt', cache=True)

modalities = [(loader, penn_treebank_transform, batcher)]

train_dataset = BaseDataset(
    source=PENN_TREEBANK_TRAIN,
    modalities=modalities
)
val_dataset = BaseDataset(
    source=PENN_TREEBANK_VALID,
    modalities=modalities
)
test_dataset = BaseDataset(
    source=PENN_TREEBANK_TEST,
    modalities=modalities
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
model = vseq.models.CWVAELM(
    num_embeddings=len(token_map),
    embedding_dim=args.embedding_dim,
    hidden_size=args.hidden_size,
    latent_dims=args.latent_dims,
    clock_rates=args.clock_rates,
    delimiter_token_idx=delimiter_token_idx,
)
model = model.to(device)
print(model)
wandb.watch(model, log='all', log_freq=len(train_loader))

x, x_sl = next(iter(train_loader))[0]
x = x.to(device)
model.summary(input_data=x, x_sl=x_sl)

prior_samples = model.prior().sample(torch.Size([args.prior_samples, 1]))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

tracker = Tracker()

beta_annealer = CosineAnnealer(n_steps=args.anneal_steps, start_value=args.anneal_start_value, end_value=1)
for epoch in tracker.epochs(args.epochs):

    model.train()
    for b, ((x, x_sl), metadata) in enumerate(tracker(train_loader)):
        x = x.to(device)

        loss, metrics, outputs = model(x, x_sl, beta=beta_annealer.value, word_dropout_rate=args.word_dropout)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)
        beta_annealer.step()

    model.eval()
    with torch.no_grad():
        for (x, x_sl), metadata in tracker(val_loader):
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl, beta=beta_annealer.value, word_dropout_rate=0.0)

            tracker.update(metrics)

        for (x, x_sl), metadata in tracker(test_loader):
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl, beta=beta_annealer.value, word_dropout_rate=0.0)

            tracker.update(metrics)

    # Get samples from prior
    (x, x_sl), log_prob = model.generate(z=prior_samples)
    text = token_map.decode_batch(x, x_sl, join_separator=" ")
    data = [(i, t) for i, t in enumerate(text)]
    prior_samples_table = wandb.Table(columns=["Idx", "Samples"], data=data)

    # Log tracker metrics
    tracker.log(
        beta=beta_annealer.value,
        samples=prior_samples_table,
        interpolations=interpolations_table,
    )
