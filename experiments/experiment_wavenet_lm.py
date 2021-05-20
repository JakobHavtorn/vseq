import os
import argparse
import logging

import torch
import torchaudio
import wandb
import rich

from torch.utils.data import DataLoader

import vseq.models

from vseq.data import BaseDataset
from vseq.data.batchers import TextBatcher
from vseq.data.datapaths import PENN_TREEBANK_TEST, PENN_TREEBANK_TRAIN, PENN_TREEBANK_VALID
from vseq.data.loaders import TextLoader
from vseq.data.tokens import DELIMITER_TOKEN, ENGLISH_STANDARD, PENN_TREEBANK_ALPHABET, UNKNOWN_TOKEN
from vseq.data.tokenizers import char_tokenizer, word_tokenizer
from vseq.data.token_map import TokenMap
from vseq.data.transforms import Compose, EncodeInteger, TextCleaner
from vseq.data.vocabulary import load_vocabulary
from vseq.utils.argparsing import str2bool
from vseq.utils.device import get_device
from vseq.utils.rand import set_seed, get_random_seed
from vseq.evaluation import Tracker



LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--layer_size", default=10, type=int, help="number of layers per stack")
parser.add_argument("--stack_size", default=4, type=int, help="number of stacks")
parser.add_argument("--res_channels", default=64, type=int, help="number of channels in residual connections")
parser.add_argument("--token_level", default="char", type=str, choices=["word", "char"], help="word- or character-level modelling")
parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
parser.add_argument("--cache_dataset", default=False, type=str2bool, help="if True, cache the dataset in RAM")
parser.add_argument("--num_workers", default=8, type=int, help="number of dataloader workers")
parser.add_argument("--wandb_group", default='lm', type=str, help="custom group for this experiment (optional)")
parser.add_argument("--seed", default=None, type=int, help="seed for random number generators. Random if -1.")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
parser.add_argument("--save_freq", default=10, type=int, help="number of epochs to go between saves")
parser.add_argument("--delete_last_model", default=False, type=str2bool, help="if True, delete the last model saved")

args, _ = parser.parse_known_args()

if args.seed is None:
    args.seed = get_random_seed()
set_seed(args.seed)

device = get_device() if args.device == "auto" else torch.device(args.device)

model_name_str = f"wavenet-lm-{args.layer_size}-{args.stack_size}-{args.res_channels}"
print(f"Initializing model with name: {model_name_str}")
wandb.init(
    entity="vseq",
    project="wavenet",
    group=args.wandb_group,
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

model = vseq.models.WaveNet(
    layer_size=args.layer_size,
    stack_size=args.stack_size,
    in_channels=len(token_map),
    res_channels=args.res_channels,
    out_classes=len(token_map),
)
model = model.to(device)
print(model)
rich.print(model.receptive_field)

batcher = TextBatcher(min_sample_length=model.receptive_field + 1)
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


wandb.watch(model, log="all", log_freq=len(train_loader))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# (x, x_sl), metadata = next(iter(train_loader))
# x = x.to(device)
# print(model.summary(input_example=x, x_sl=x_sl))


tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker.steps(train_loader):
        x = x.to(device)

        loss, metrics, output = model(x, x_sl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.update(metrics)

    model.eval()
    with torch.no_grad():
        for (x, x_sl), metadata in tracker.steps(val_loader):
            x = x.to(device)

            loss, metrics, output = model(x, x_sl)

            tracker.update(metrics)

        tracker.log()

        # # save samples
        # x = model.generate(n_samples=32, n_frames=96000)
        # x = x.unsqueeze(-1).to(torch.uint8).cpu()
        # for i in range(len(x)):
        #     torchaudio.save(
        #         f"./wavenet_samples/{model_name_str}-epoch-{epoch}-sample_{i}.wav",
        #         x[i],
        #         sample_rate=16000,
        #         channels_first=False,
        #         encoding="ULAW",
        #     )
    if epoch != 0 and epoch % args.save_freq == 0:  # epoch k, 2k, 3k, ...
        if args.delete_last_model and os.path.exists(f"./models/{model_name_str}-epoch-{epoch-args.save_freq}"):
            # delete past model 
            os.removedirs(f"./models/{model_name_str}-epoch-{epoch-args.save_freq}")
        model.save(f"./models/{model_name_str}-epoch-{epoch}")
