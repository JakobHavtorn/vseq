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
from vseq.data.batchers import SpectrogramBatcher
from vseq.data.datapaths import TIMIT_TEST, TIMIT_TRAIN
from vseq.data.loaders import AudioLoader
from vseq.data.transforms import StackWaveform
from vseq.evaluation import Tracker
from vseq.utils.argparsing import str2bool
from vseq.utils.rand import set_seed, get_random_seed
from vseq.training.annealers import CosineAnnealer


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
parser.add_argument("--stack_frames", default=200, type=int, help="Number of audio frames to stack in feature vector")
parser.add_argument("--hidden_size", default=[512, 512, 512], type=int, nargs="+", help="dimensionality of hidden state in CWVAE")
parser.add_argument("--latent_size", default=[128, 128, 128], type=int, nargs="+", help="dimensionality of latent state in CWVAE")
parser.add_argument("--time_factors", default=6, type=int, nargs="+", help="temporal abstraction factor")
parser.add_argument("--n_dense", default=3, type=int, help="dense layers for embedding per level")
parser.add_argument("--beta_anneal_steps", default=0, type=int, help="number of steps to anneal beta")
parser.add_argument("--beta_start_value", default=0, type=float, help="initial beta annealing value")
parser.add_argument("--free_nats_steps", default=0, type=int, help="number of steps to constant/anneal free bits")
parser.add_argument("--free_nats_start_value", default=1, type=float, help="free bits per timestep")
parser.add_argument("--epochs", default=750, type=int, help="number of epochs")
parser.add_argument("--num_workers", default=4, type=int, help="number of dataloader workers")
parser.add_argument("--seed", default=None, type=int, help="random seed")
parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

args = parser.parse_args()


if args.seed is None:
    args.seed = get_random_seed()

set_seed(args.seed)

device = vseq.utils.device.get_device() if args.device == "auto" else torch.device(args.device)


wandb.init(
    entity="vseq",
    project="cwvae",
    group=None,
)
wandb.config.update(args)
rich.print(vars(args))


# loader = AudioLoader("wav", cache=False)
# batcher = ListBatcher()
# mean, variance = BaseDataset(source=TIMIT_TRAIN, modalities=[(loader, None, batcher)], sort=False).compute_statistics(
#     num_workers=args.num_workers
# )

# batcher = SpectrogramBatcher()
# transform = Compose(Normalize(mean=mean, std=math.sqrt(variance)), StackWaveform(args.stack_frames))

loader = AudioLoader("wav", cache=False)
batcher = SpectrogramBatcher()
transform = StackWaveform(args.stack_frames)
modalities = [(loader, transform, batcher)]

train_dataset = BaseDataset(
    source=TIMIT_TRAIN,
    modalities=modalities,
)
test_dataset = BaseDataset(
    source=TIMIT_TEST,
    modalities=modalities,
)

train_loader = DataLoader(
    dataset=train_dataset,
    collate_fn=train_dataset.collate,
    num_workers=args.num_workers,
    shuffle=True,
    batch_size=args.batch_size,
    pin_memory=True,
    drop_last=True,
)
test_loader = DataLoader(
    dataset=test_dataset,
    collate_fn=test_dataset.collate,
    num_workers=args.num_workers,
    shuffle=False,
    batch_size=args.batch_size,
    pin_memory=True,
)


model = vseq.models.CWVAEAudioStacked(
    input_size=args.stack_frames,
    z_size=args.latent_size,
    h_size=args.hidden_size,
    time_factors=args.time_factors,
    n_dense=args.n_dense,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()

        tracker.update(metrics)

    model.eval()
    with torch.no_grad():
        for (x, x_sl), metadata in tracker(test_loader):
            x = x.to(device)

            loss, metrics, outputs = model(x, x_sl)

            tracker.update(metrics)

        reconstructions = [wandb.Audio(outputs.x_hat[i].flatten().cpu().numpy(), caption=f"Reconstruction {i}", sample_rate=16000) for i in range(2)]

        # (x, x_sl), outputs = model.generate(n_samples=2, max_timesteps=128000 // args.stack_frames)
        # samples = [wandb.Audio(x[i].flatten().cpu().numpy(), caption=f"Sample {i}", sample_rate=16000) for i in range(2)]

    # tracker.log(samples=samples, reconstructions=reconstructions)
    tracker.log(reconstructions=reconstructions)
    # tracker.log()
