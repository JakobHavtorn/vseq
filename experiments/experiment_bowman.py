import argparse
import logging
import tqdm

import torch
import torchvision

from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

import vseq
import vseq.data
import vseq.models
import vseq.training
import vseq.utils
import vseq.utils.device

from vseq.data import transforms
from vseq.data import DataModule, BaseDataset
from vseq.data.collate import collate_spectrogram, collate_text
from vseq.data.transforms import EncodeInteger, Compose
from vseq.data.datapaths import LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_TRAIN
from vseq.data.text_cleaners import clean_librispeech
from vseq.data.tokens import LIBRISPEECH_TOKENS
from vseq.data.tokenizers import char_tokenizer
from vseq.data.token_map import TokenMap
from vseq.data.samplers import EvalSampler, FrameSampler


LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="VAE", help="model type (vae | lvae | biva)")
parser.add_argument("--epochs", default=500, type=int, help="number of epochs")
parser.add_argument("--lr", default=2e-3, type=float, help="base learning rate")
parser.add_argument("--test_every", default=1, type=int, help="test every x epochs")
parser.add_argument("--seed", default=42, type=int, help="random seed")
parser.add_argument("--task_name", default="", type=str, help="run task_name suffix")
parser.add_argument("--task_tags", default=[], type=str, nargs="+", help="tags for the task")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args, _ = parser.parse_known_args()


device = vseq.utils.device.get_device() if args.device == "auto" else torch.device(args.device)

token_map = TokenMap(tokens=LIBRISPEECH_TOKENS)
LibrispeecTextTransform = transforms.Compose(
    transforms.TextCleaner(clean_librispeech),
    EncodeInteger(
        token_map=token_map,
        tokenizer=char_tokenizer
    ),
)

modalities = [
    # ('flac', MelSpectrogram(), collate_spectrogram),
    ('txt', LibrispeecTextTransform, collate_text)
]


train_dataset = BaseDataset(
    source=LIBRISPEECH_TRAIN,
    modalities=modalities,
)
val_dataset = BaseDataset(
    source=LIBRISPEECH_DEV_CLEAN,
    modalities=modalities,
)

train_sampler = FrameSampler(source=LIBRISPEECH_TRAIN, sample_rate=16000, max_seconds=320)
val_sampler = EvalSampler(source=LIBRISPEECH_DEV_CLEAN, sample_rate=16000, max_seconds=320)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=4,
    batch_sampler=train_sampler
)
val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=4,
    batch_sampler=val_sampler
)

model = vseq.models.Bowman(num_embeddings=len(token_map), embedding_dim=16, hidden_size=64)
model = model.to(device)

criterion = lambda *x: x



for epoch in range(args.epochs):
    for (x, x_sl), metadata in tqdm.tqdm(train_loader, total=len(train_loader)):
        # import IPython; IPython.embed()

        x = x.to(device)

        x_hat = model(x, x_sl)

        loss = criterion(x, x_hat, x_sl)

        # loss.backward()




