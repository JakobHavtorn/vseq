import argparse
import logging

import torch
import torchvision

from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

import vseq
import vseq.data
from vseq.data import collate
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

model = vseq.models.VAE()


LibrispeecTextTransform = transforms.Compose(
    transforms.TextCleaner(clean_librispeech),
    EncodeInteger(
        token_map=TokenMap(tokens=LIBRISPEECH_TOKENS),
        tokenizer=char_tokenizer
    ),
)

modalities = [
    ('flac', MelSpectrogram(), collate_spectrogram),
    ('txt', LibrispeecTextTransform, collate_text)
]


train_dataset = BaseDataset(
    source=LIBRISPEECH_DEV_CLEAN,
    modalities=modalities,
)


batch = [train_dataset[i] for i in range(32)]

outputs, metadata = train_dataset.collate(batch)

import IPython; IPython.embed(using=False)

dataloader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    batch_size=64,
    num_workers=4,
    shuffle=True,
    sampler=None
)



import time
import tqdm

t_start = time.time()
for ((x, x_sl), (y, y_sl)), (m_x, m_y) in tqdm.tqdm(dataloader, total=len(dataloader)):
    pass

print(time.time() - t_start)

import IPython; IPython.embed(using=False)
