import os 
import argparse
import logging

import torch
import torchaudio
import numpy as np
import wandb
import rich
import matplotlib.pyplot as plt



from torch.utils.data import DataLoader

import vseq.models
from vseq.models.base_model import load_model


from vseq.data import BaseDataset
from vseq.data.batchers import AudioBatcher
from vseq.data.datapaths import LIBRISPEECH_DEV_CLEAN, LIBRISPEECH_TRAIN
from vseq.data.loaders import AudioLoader
from vseq.data.transforms import Compose, Quantize, RandomSegment, Scale, MuLawEncode
from vseq.evaluation.tracker import Tracker
from vseq.utils.argparsing import str2bool
from vseq.utils.device import get_device
from vseq.utils.rand import set_seed, get_random_seed


LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, help="model path to load")
parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument("--input_coding", default="mu_law", type=str, choices=["mu_law", "frames"], help="input encoding")
parser.add_argument("--epochs", default=5, type=int, help="number of epochs")
parser.add_argument("--cache_dataset", default=False, type=str2bool, help="if True, cache the dataset in RAM")
parser.add_argument("--num_workers", default=8, type=int, help="number of dataloader workers")
parser.add_argument("--seed", default=None, type=int, help="seed for random number generators. Random if -1.")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args, _ = parser.parse_known_args()

if args.seed is None:
    args.seed = get_random_seed()
set_seed(args.seed)

device = get_device() if args.device == "auto" else torch.device(args.device)


if args.input_coding == "mu_law":
    args.in_channels = 256
elif args.input_coding == "frames":
    args.in_channels = 1
else:
    raise ValueError()


rich.print(vars(args))



# Load model

assert os.path.exists(args.model_path)

model = load_model(args.model_path)
model = model.to(device)
print(model)
print(model.receptive_field)



if args.input_coding == "mu_law":
    wavenet_transform = Compose(RandomSegment(length=model.receptive_field+1), MuLawEncode(), Quantize(bits=8))
elif args.input_coding == "frames":
    wavenet_transform = Compose(RandomSegment(length=model.receptive_field+1))


modalities = [
    (AudioLoader("flac"), wavenet_transform, AudioBatcher()),
]

val_dataset = BaseDataset(
    source=LIBRISPEECH_DEV_CLEAN,
    modalities=modalities,
)

val_loader = DataLoader(
    dataset=val_dataset,
    collate_fn=val_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size,
    pin_memory=True,
)


# (x, x_sl), metadata = next(iter(train_loader))
# x = x.to(device)
# print(model.summary(input_example=x, x_sl=x_sl))


tracker = Tracker()

grad_tensors = []

for epoch in tracker.epochs(args.epochs):

    grad_storage = torch.empty((len(val_loader)+1)*args.batch_size, model.receptive_field+1)

    for i, ((x, x_sl), metadata) in enumerate(tracker.steps(val_loader)):
        x=x.to(device)

        loss, metrics, output = model(x, x_sl)

        loss.backward()
        x_out = output.x
        x_out.grad.shape # should be B, C, T

        grad_storage[i*args.batch_size:i*args.batch_size + x_out.shape[0]] = torch.linalg.norm(x_out.detach(), dim=1)

    grad_tensors.append(grad_storage)



grads = torch.cat(grad_tensors, 0)
grads.shape

grad_mean = grads.mean(axis=0).numpy()
grad_var = grads.var(axis=0).numpy()
frame_idx = np.arange(-model.receptive_field, 1)



fig, ax = plt.subplots()

ax.plot(frame_idx,grad_mean, "k")

ax.fill_between(frame_idx,grad_mean - grad_var,grad_mean + grad_var,
    alpha=0.5, facecolor='#FF6432')

ax.set_title(args.model_path.split("/")[-1] + f" RF: {model.receptive_field}")
ax.set_ylabel("Mean +/- STD of gradient")
ax.set_xlabel("Time index of input")

fig.savefig(f"misc/plots/{args.model_path.split('/')[-1]}-gradients.png")