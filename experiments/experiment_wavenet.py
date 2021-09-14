import argparse
import logging
import os

import torch
import wandb
import rich

from torch.utils.data import DataLoader

import vseq.models

from vseq.data import BaseDataset
from vseq.data.batchers import AudioBatcher
from vseq.data.datapaths import DATASETS, TIMIT_TRAIN, TIMIT_TEST
from vseq.data.loaders import AudioLoader
from vseq.data.samplers.batch_samplers import LengthEvalSampler, LengthTrainSampler
from vseq.data.transforms import Compose, MuLawDecode, Quantize, RandomSegment, MuLawEncode, StackWaveform
from vseq.evaluation.tracker import Tracker
from vseq.modules.distributions import CategoricalDense, DiscretizedLogisticMixtureDense
from vseq.utils.argparsing import str2bool
from vseq.utils.device import get_device
from vseq.utils.rand import set_seed, get_random_seed


LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=0, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--n_layers", default=10, type=int, help="number of layers per stack")
parser.add_argument("--n_stacks", default=4, type=int, help="number of stacks")
parser.add_argument("--res_channels", default=64, type=int, help="number of channels in residual connections")
parser.add_argument("--input_coding", default="mu_law", type=str, choices=["mu_law", "frames"], help="input encoding")
parser.add_argument("--input_embedding_dim", default=1, type=int, help="if not 1, embed frames (after potential mu-law) as vector of this dimension")
parser.add_argument("--num_bits", default=16, type=int, help="number of bits for mu_law encoding (note the data bits depth)")
parser.add_argument("--distribution", default="dmol", type=str, help="distribution for the output p(x_t|x_t-1)")
parser.add_argument("--stack_frames", default=1, type=int, help="Number of audio frames to stack in feature vector if input_coding is frames")
parser.add_argument("--epochs", default=3000, type=int, help="number of epochs")
parser.add_argument("--cache_dataset", default=False, type=str2bool, help="if True, cache the dataset in RAM")
parser.add_argument("--num_workers", default=8, type=int, help="number of dataloader workers")
parser.add_argument("--dataset", default="timit", type=str, help="dataset to train on")
parser.add_argument("--wandb_group", default=None, type=str, help="custom group for this experiment (optional)")
parser.add_argument("--save_checkpoints", default=False, type=str2bool, help="whether to store checkpoints or not")
parser.add_argument("--seed", default=None, type=int, help="seed for random number generators. Random if -1.")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
parser.add_argument("--use_amp", action="store_true", help="if true, use automatic mixed precision")

args = parser.parse_args()

if args.seed is None:
    args.seed = get_random_seed()
set_seed(args.seed)

device = get_device() if args.device == "auto" else torch.device(args.device)

dataset = DATASETS[args.dataset]


wandb.init(
    entity="vseq",
    project="wavenet",
    group=None,
)
wandb.config.update(args)
rich.print(vars(args))


if args.distribution == "dmol":
    likelihood = DiscretizedLogisticMixtureDense(args.res_channels, 1, num_mix=10, num_bins=2**args.num_bits)
elif args.distribution == "categorical":
    likelihood = CategoricalDense(args.res_chanels, 2**args.num_bits)
else:
    raise ValueError(f"Unknown distribution: {args.distribution}")

encode_transform = []
decode_transform = []
if args.input_coding == "mu_law":
    encode_transform.append(MuLawEncode(bits=args.num_bits))
    decode_transform.append(MuLawDecode(bits=args.num_bits))

if args.input_embedding_dim > 1:
    encode_transform.append(Quantize(bits=args.num_bits))

if args.stack_frames > 1:
    encode_transform.append(StackWaveform(n_frames=args.stack_frames))

if args.distribution == "categorical":
    decode_transform.append()  # Opposite of Quantize operation (Scale?)

encode_transform = Compose(*encode_transform)
decode_transform = Compose(*decode_transform)

modalities = [(AudioLoader("wav"), encode_transform, AudioBatcher())]


train_sampler = LengthTrainSampler(
    source=dataset.train,
    field="length.wav.samples",
    max_len=16000 * args.batch_size if args.batch_size > 0 else "max",
    max_pool_difference=16000 * 0.3,
    min_pool_size=512,
    # num_batches=784
)
valid_sampler = LengthEvalSampler(
    source=dataset.test,
    field="length.wav.samples",
    max_len=16000 * args.batch_size if args.batch_size > 0 else "max",
)
# train_sampler.batches = train_sampler.batches[:3]
# valid_sampler.batches = valid_sampler.batches[:3]
train_dataset = BaseDataset(
    source=dataset.train,
    modalities=modalities,
)
valid_dataset = BaseDataset(
    source=dataset.test,
    modalities=modalities,
)
train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    batch_sampler=train_sampler,
    pin_memory=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    collate_fn=valid_dataset.collate,
    num_workers=args.num_workers,
    batch_sampler=valid_sampler,
    pin_memory=True,
)

model = vseq.models.WaveNet(
    likelihood=likelihood,
    n_layers=args.n_layers,
    n_stacks=args.n_stacks,
    in_channels=args.input_embedding_dim,
    res_channels=args.res_channels,
    num_bins=2 ** args.num_bits,
)

(x, x_sl), metadata = next(iter(train_loader))
model.summary(input_data=x, x_sl=x_sl)
model = model.to(device)
# print(model)
rich.print(model.receptive_field)
wandb.watch(model, log="all", log_freq=len(train_loader))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

tracker = Tracker()

for epoch in tracker.epochs(args.epochs):

    model.train()
    for (x, x_sl), metadata in tracker.steps(train_loader):
        x = x.to(device, non_blocking=True)

        # TODO If categorical target is different from x (i.e. quantized, ints)
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            loss, metrics, output = model(x, x_sl)

        optimizer.zero_grad(set_to_none=True)
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tracker.update(metrics)

    model.eval()
    with torch.no_grad():
        for (x, x_sl), metadata in tracker.steps(valid_loader):
            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                loss, metrics, output = model(x, x_sl)

            tracker.update(metrics)

        extra = dict()
        if epoch % 25 == 0:
            predictions = decode_transform(output.predictions)
            predictions = [wandb.Audio(predictions[i].cpu().flatten().numpy(), caption=f"Reconstruction {i}", sample_rate=16000) for i in range(min(predictions.shape[0], 2))]

            x = model.generate(n_samples=2, n_frames=128000 // args.stack_frames)
            x = decode_transform(x)
            samples = [wandb.Audio(x[i].flatten().cpu().numpy(), caption=f"Sample {i}", sample_rate=16000) for i in range(2)]

            extra = dict(samples=samples, predictions=predictions)

        if (
            args.save_checkpoints
            and wandb.run is not None and wandb.run.dir != "/"
            and epoch > 1
            and min(tracker.accumulated_values[dataset.test]["loss"][:-1])
            > tracker.accumulated_values[dataset.test]["loss"][-1]
        ):
            model.save(wandb.run.dir)
            checkpoint = dict(
                epoch=epoch,
                best_loss=tracker.accumulated_values[dataset.test]["loss"][-1],
                optimizer_state_dict=optimizer.state_dict(),
            )
            torch.save(checkpoint, os.path.join(wandb.run.dir, "checkpoint.pt"))
            print(f"Saved model checkpoint at {wandb.run.dir}")

        tracker.log(**extra)
