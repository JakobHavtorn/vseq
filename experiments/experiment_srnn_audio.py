import argparse
import os

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
from vseq.data.batchers import ListBatcher, SpectrogramBatcher
from vseq.data.datapaths import DATASETS, TIMIT_TEST, TIMIT_TRAIN
from vseq.data.loaders import AudioLoader
from vseq.data.samplers.batch_samplers import LengthEvalSampler, LengthTrainSampler
from vseq.data.transforms import Compose, MuLawDecode, MuLawEncode, Normalize, StackWaveform
from vseq.evaluation import Tracker
from vseq.utils.argparsing import str2bool
from vseq.utils.rand import set_seed, get_random_seed
from vseq.training.annealers import CosineAnnealer


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--stack_frames", default=200, type=int, help="Number of audio frames to stack in feature vector")
parser.add_argument("--hidden_size", default=512, type=int, help="dimensionality of hidden state in VRNN")
parser.add_argument("--latent_size", default=128, type=int, help="dimensionality of latent state in VRNN")
parser.add_argument("--residual_posterior", default=False, type=str2bool, help="residual parameterization of posterior")
parser.add_argument("--word_dropout", default=0.0, type=float, help="word dropout")
parser.add_argument("--dropout", default=0.0, type=float, help="dropout")
parser.add_argument("--beta_anneal_steps", default=0, type=int, help="number of steps to anneal beta")
parser.add_argument("--beta_start_value", default=0, type=float, help="initial beta annealing value")
parser.add_argument("--free_nats_steps", default=0, type=int, help="number of steps to constant/anneal free bits")
parser.add_argument("--free_nats_start_value", default=8, type=float, help="free bits per timestep")
parser.add_argument("--input_coding", default="mu_law", type=str, choices=["mu_law", "frames"], help="input encoding")
parser.add_argument("--num_bits", default=8, type=int, help="number of bits for DML and input")
parser.add_argument("--num_mix", default=10, type=int, help="number of logistic mixture components")
parser.add_argument("--dataset", default="timit", type=str, help="dataset to use")
parser.add_argument("--epochs", default=750, type=int, help="number of epochs")
parser.add_argument("--num_workers", default=4, type=int, help="number of dataloader workers")
parser.add_argument("--save_checkpoints", default=False, type=str2bool, help="whether to store checkpoints or not")
parser.add_argument("--seed", default=None, type=int, help="random seed")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args = parser.parse_args()


if args.seed is None:
    args.seed = get_random_seed()

set_seed(args.seed)

device = vseq.utils.device.get_device() if args.device == "auto" else torch.device(args.device)

dataset = DATASETS[args.dataset]


wandb.init(
    entity="vseq",
    project="srnn",
    group=None,
)
wandb.config.update(args)
rich.print(vars(args))


if args.input_coding == "mu_law":
    encode_transform = Compose(MuLawEncode(bits=args.num_bits), StackWaveform(n_frames=args.stack_frames))
    decode_transform = Compose(torch.nn.Flatten(start_dim=1), MuLawDecode(bits=args.num_bits))
else:
    encode_transform = StackWaveform(n_frames=args.stack_frames)
    decode_transform = torch.nn.Flatten(start_dim=1)

loader = AudioLoader("wav", cache=False)
batcher = SpectrogramBatcher()
modalities = [(loader, encode_transform, batcher)]


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


model = vseq.models.SRNNAudioDML(
    input_size=args.stack_frames,
    hidden_size=args.hidden_size,
    latent_size=args.latent_size,
    word_dropout=args.word_dropout,
    dropout=args.dropout,
    residual_posterior=args.residual_posterior,
    num_mix=args.num_mix,
    num_bins=2 ** args.num_bits,
)

print(model)
x, x_sl = next(iter(train_loader))[0]
model.summary(input_data=x[:, :1], x_sl=torch.LongTensor([1] * x.size(0)), device='cpu')
model = model.to(device)
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
        for (x, x_sl), metadata in tracker(valid_loader):
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl)

            tracker.update(metrics)

        # Log reconstructions and samples
        extra = dict()
        if epoch % 25 == 0:
            reconstructions = decode_transform(outputs.reconstructions)
            reconstructions = [
                wandb.Audio(
                    reconstructions[i].flatten().cpu().numpy(), caption=f"Reconstruction {i}", sample_rate=16000
                )
                for i in range(min(2, reconstructions.shape[0]))
            ]

            (x, x_sl), outputs = model.generate(n_samples=2, max_timesteps=128000 // args.stack_frames)
            x = decode_transform(x)
            samples = [wandb.Audio(x[i].flatten().cpu().numpy(), caption=f"Sample {i}", sample_rate=16000) for i in range(2)]
            
            extra = dict(samples=samples, reconstructions=reconstructions)

    # Save model checkpoints
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
