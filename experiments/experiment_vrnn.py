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
from vseq.data.tokens import DELIMITER_TOKEN, PENN_TREEBANK_ALPHABET, UNKNOWN_TOKEN
from vseq.data.tokenizers import char_tokenizer, word_tokenizer
from vseq.data.token_map import TokenMap
from vseq.data.transforms import Compose, EncodeInteger, TextCleaner
from vseq.data.vocabulary import load_vocabulary
from vseq.evaluation import Tracker
from vseq.utils.rand import set_seed, get_random_seed
from vseq.training.annealers import CosineAnnealer


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--embedding_dim", default=300, type=int, help="dimensionality of embedding space")
parser.add_argument("--hidden_size", default=512, type=int, help="dimensionality of hidden state in VRNN")
parser.add_argument("--latent_size", default=128, type=int, help="dimensionality of latent state in VRNN")
parser.add_argument("--word_dropout", default=0.0, type=float, help="word dropout")
parser.add_argument("--beta_anneal_steps", default=0, type=int, help="number of steps to anneal beta")
parser.add_argument("--beta_start_value", default=0, type=float, help="initial beta annealing value")
parser.add_argument("--free_nats_steps", default=0, type=int, help="number of steps to constant/anneal free bits")
parser.add_argument("--free_nats_start_value", default=8, type=float, help="free bits per timestep")
parser.add_argument("--token_level", default="word", type=str, choices=["word", "char"], help="word or character level")
parser.add_argument("--epochs", default=250, type=int, help="number of epochs")
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
    project="vrnn",
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
        EncodeInteger(token_map=token_map, tokenizer=char_tokenizer),
    )

batcher = TextBatcher()
loader = TextLoader("txt", cache=True)

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
model = vseq.models.VRNNLM(
    num_embeddings=len(token_map),
    embedding_dim=args.embedding_dim,
    hidden_size=args.hidden_size,
    latent_size=args.latent_size,
    word_dropout=args.word_dropout,
    delimiter_token_idx=delimiter_token_idx,
)

print(model)
model = model.to(device)
x, x_sl = next(iter(train_loader))[0]
x = x.to(device)
# model.summary(input_data=x[:, :1], x_sl=torch.LongTensor([1] * x.size(0)))
wandb.watch(model, log="all", log_freq=len(train_loader))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

tracker = Tracker()

beta_annealer = CosineAnnealer(anneal_steps=args.beta_anneal_steps, start_value=args.beta_start_value, end_value=1)
free_nats_annealer = CosineAnnealer(
    anneal_steps=args.free_nats_steps // 2,
    constant_steps=args.free_nats_steps // 2,
    start_value=args.free_nats_start_value,
    end_value=0,
)

for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker(train_loader):
        x = x.to(device)

        loss, metrics, outputs = model(x, x_sl, beta=beta_annealer.step(), free_nats=free_nats_annealer.step())

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

        # Log samples from prior
        (x, x_sl), outputs = model.generate(n_samples=4)
        text = token_map.decode_batch(x, x_sl, join_separator=" " if args.token_level == "word" else "")
        data = [(i, t) for i, t in enumerate(text)]
        prior_samples_table = wandb.Table(columns=["Idx", "Samples"], data=data)
        rich.print(text)

    tracker.log(samples=prior_samples_table)
