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
from vseq.data.tokens import DELIMITER_TOKEN
from vseq.data.tokenizers import word_tokenizer
from vseq.data.token_map import TokenMap
from vseq.data.transforms import EncodeInteger
from vseq.data.vocabulary import load_vocabulary
from vseq.evaluation import Tracker
from vseq.utils.argparsing import str2bool
from vseq.utils.rand import set_seed, get_random_seed
from vseq.training import CosineAnnealer


LOGGER = logging.getLogger(name=__file__)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--embedding_dim", default=353, type=int, help="dimensionality of embedding space")
parser.add_argument("--hidden_size", default=191, type=int, help="dimensionality of hidden state in LSTM")
parser.add_argument("--latent_dim", default=13, type=int, help="dimensionality of latent space")
parser.add_argument("--n_highway_blocks", default=4, type=int, help="number of Highway network blocks")
parser.add_argument("--word_dropout", default=0.38, type=float, help="word dropout probability")
parser.add_argument("--random_prior_variance", default=False, type=str2bool, help="randomly initialize prior variance")
parser.add_argument("--trainable_prior", default=False, type=str2bool, help="make prior trainable")
parser.add_argument("--anneal_steps", default=20000, type=int, help="number of steps to anneal beta")
parser.add_argument("--anneal_start_value", default=0, type=float, help="initial beta annealing value")
parser.add_argument("--prior_samples", default=32, type=int, help="number of prior samples for logging")
parser.add_argument("--n_interpolations", default=10, type=int, help="number of interpolation samples for logging")
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
    project="bowman",
    group=None,
)
wandb.config.update(args)
rich.print(vars(args))


vocab = load_vocabulary(PENN_TREEBANK_TRAIN)
token_map = TokenMap(tokens=vocab, add_start=False, add_end=False, add_delimit=True)
penn_treebank_transform = EncodeInteger(
    token_map=token_map,
    tokenizer=word_tokenizer,
)
batcher = TextBatcher()
loader = TextLoader('txt', cache=True)

modalities = [(loader, penn_treebank_transform, batcher)]

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
model = vseq.models.Bowman(
    num_embeddings=len(token_map),
    embedding_dim=args.embedding_dim,
    hidden_size=args.hidden_size,
    latent_dim=args.latent_dim,
    n_highway_blocks=args.n_highway_blocks,
    delimiter_token_idx=delimiter_token_idx,
    random_prior_variance=args.random_prior_variance,
    trainable_prior=args.trainable_prior,
)
model = model.to(device)
print(model)
wandb.watch(model, log='all', log_freq=len(train_loader))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

x, x_sl = next(iter(train_loader))[0]
x = x.to(device)
print(model.summary(input_example=x, x_sl=x_sl))


prior_samples = model.prior().sample(torch.Size([args.prior_samples, 1]))

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

    # Perform interpolation in latent space
    n_interps = args.n_interpolations
    x = ["she did n't want to be with him", "i want to talk to you"]
    x = [penn_treebank_transform(_x) for _x in x]
    x, x_sl = batcher(x)

    _, q_z = model.infer(x.to(device), x_sl)
    z_samples = q_z.mean.unsqueeze(-1).repeat(1, 1, 1, n_interps)  # Create interpolation axis
    alpha = torch.linspace(0, 1, n_interps).to(z_samples.device)
    z_interps = z_samples[0] * (1 - alpha) + z_samples[1] * alpha
    z_interps = z_interps.permute(2, 0, 1)
    (x, x_sl), log_prob = model.generate(z=z_interps, use_mode=True)

    text = token_map.decode_batch(x, x_sl, join_separator=" ")
    data = [(i, t) for i, t in enumerate(text)]
    interpolations_table = wandb.Table(columns=["Idx", "Samples"], data=data)

    # Log tracker metrics
    tracker.log(
        beta=beta_annealer.value,
        samples=prior_samples_table,
        interpolations=interpolations_table,
    )
