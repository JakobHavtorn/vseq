import argparse
import logging

import torch

from torchaudio.transforms import MelSpectrogram

import vseq
import vseq.data
import vseq.models
import vseq.training
import vseq.utils
import vseq.utils.device

from vseq.data import DataModule, BaseDataset
from vseq.data.collate import collate_spectrogram, collate_text
from vseq.data.transforms import EncodeInteger
from vseq.data.datapaths import LIBRISPEECH_TRAIN


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

modalities = [
    ('wav', MelSpectrogram(), collate_spectrogram),
    ('txt', EncodeInteger(), collate_text)
]


train_dataset = BaseDataset(
    source=LIBRISPEECH_TRAIN,
    modalities=modalities,
)

import IPython; IPython.embed()

