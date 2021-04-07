import argparse
import logging

import torch
import wandb

from torch.utils.data import DataLoader

import vseq
import vseq.data
import vseq.models
import vseq.utils
import vseq.utils.device

from vseq.data import DataModule, BaseDataset
from vseq.data.batcher import TextBatcher
from vseq.data.datapaths import PENN_TREEBANK_VALID, PENN_TREEBANK_TRAIN, PENN_TREEBANK_TRAIN, PENN_TREEBANK_VALID
from vseq.evaluation.metrics import BitsPerDimMetric
from vseq.data.tokens import  DELIMITER_TOKEN
from vseq.data.tokenizers import word_tokenizer
from vseq.data.token_map import TokenMap
from vseq.data.transforms import EncodeInteger
from vseq.data.vocabulary import load_vocabulary
from vseq.evaluation import LLMetric, KLMetric, PerplexityMetric, Tracker
from vseq.utils.rand import set_seed


torch.autograd.set_detect_anomaly(True)
LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_dim", default=353, type=int, help="dimensionality of embedding space")
parser.add_argument("--hidden_size", default=191, type=int, help="dimensionality of hidden state in LSTM")
parser.add_argument("--latent_dim", default=13, type=int, help="dimensionality of latent space")
parser.add_argument("--word_dropout", default=0.38, type=float, help="word dropout probability")
parser.add_argument("--epochs", default=500, type=int, help="number of epochs")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--batch_size", default=32, type=int, help="batch size")
parser.add_argument("--num_workers", default=2, type=int, help="number of dataloader workers")
parser.add_argument("--seed", default=42, type=int, help="random seed")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args, _ = parser.parse_known_args()

wandb.init(
    project='vseq',
    group='bowman',
    # name='original',
)
wandb.config.update(args)

set_seed(args.seed)

device = vseq.utils.device.get_device() if args.device == "auto" else torch.device(args.device)

vocab = load_vocabulary(PENN_TREEBANK_TRAIN)
token_map = TokenMap(tokens=vocab, add_start=False, add_end=False, add_delimit=True)
penn_treebank_transform = EncodeInteger(
    token_map=token_map,
    tokenizer=word_tokenizer,
)
batcher = TextBatcher()

modalities = [
    ("txt", penn_treebank_transform, batcher)
]

train_dataset = BaseDataset(
    source=PENN_TREEBANK_TRAIN,
    modalities=modalities,
    cache=True,
)
val_dataset = BaseDataset(
    source=PENN_TREEBANK_VALID,
    modalities=modalities,
    cache=True,
)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    shuffle=True,
    batch_size=args.batch_size
)
val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size
)

delimiter_token_idx = token_map.get_index(DELIMITER_TOKEN)
model = vseq.models.Bowman(
    num_embeddings=len(token_map),
    embedding_dim=args.embedding_dim,
    hidden_size=args.hidden_size,
    latent_dim=args.latent_dim,
    delimiter_token_idx=delimiter_token_idx
)

wandb.watch(model)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
print(model)


# x, x_sl = next(iter(train_loader))[0]
# x = x.to(device)
# x_sl = x_sl.to(device)
# model.summary(x.shape, batch_size=1, x_sl=x_sl)


metric_loss = LLMetric(name='loss', tags={'loss'})
metric_elbo = LLMetric(name='elbo')
metric_rec = LLMetric(name='rec')
metric_kl = KLMetric()
metric_pp = PerplexityMetric()
metric_bpd = BitsPerDimMetric()

tracker = Tracker(metric_rec, metric_kl, metric_elbo, metric_pp, metric_bpd)


# elbo_train = []
# elbo = []
for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker(train_loader):
        x = x.to(device)

        loss, output = model(x, x_sl, word_dropout_rate=args.word_dropout)  # TODO Fix word dropout error

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_loss.update(loss, weight_by=output.elbo.numel())
        metric_elbo.update(output.elbo)
        metric_rec.update(output.rec)
        metric_kl.update(output.kl)
        metric_bpd.update(output.elbo, reduce_by=x_sl - 1)
        metric_pp.update(output.elbo, reduce_by=x_sl - 1)

        # elbo_train.append(loss.item())

    model.eval()
    for (x, x_sl), metadata in tracker(val_loader):
        x = x.to(device)

        loss, output = model(x, x_sl, word_dropout_rate=0.0)

        metric_loss.update(loss, weight_by=output.elbo.numel())
        metric_elbo.update(output.elbo)
        metric_rec.update(output.rec)
        metric_kl.update(output.kl)
        metric_bpd.update(output.elbo, reduce_by=x_sl - 1)
        metric_pp.update(output.elbo, reduce_by=x_sl - 1)

        # elbo.append(loss.item())


    # model.generate(n_samples=)

    # import matplotlib.pyplot as plt

    # plt.plot(elbo_train)
    # plt.savefig('batch_elbo_train.pdf')
    # plt.cla()

    # plt.plot(elbo)
    # plt.savefig('batch_elbo.pdf')
    # plt.cla()

    wandb.log({'elbo_valid': getattr(tracker.sources, val_dataset.source).elbo})
    tracker.log()
