import argparse
import os
from vseq.data.samplers.distributed_sampler import DistributedSamplerWrapper

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as distributed
import wandb
import rich

from torch.utils.data import DataLoader

import vseq
import vseq.data
import vseq.models
import vseq.utils
import vseq.utils.device

from vseq.data import BaseDataset
from vseq.data.batchers import AudioBatcher, SpectrogramBatcher
from vseq.data.datapaths import TIMIT_TEST, TIMIT_TRAIN
from vseq.data.loaders import AudioLoader
from vseq.data.transforms import Compose, MuLawDecode, MuLawEncode, Quantize, StackWaveform
from vseq.evaluation import Tracker
from vseq.utils.argparsing import str2bool
from vseq.utils.rand import set_seed, get_random_seed
from vseq.training.annealers import CosineAnnealer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp_master_addr", default="localhost", type=str, help="address for the DDP master")
    parser.add_argument("--ddp_master_port", default="123456", type=str, help="port for the DDP master")
    parser.add_argument("--nodes", "-n", default=1, type=int, help="number of nodes")
    parser.add_argument("--gpus", "-g", default=1, type=int, help="number of gpus per node")
    parser.add_argument("--node_rank", "-nr", default=0, type=int, help="ranking of this node within the nodes")

    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--lr", default=3e-4, type=float, help="base learning rate")
    parser.add_argument("--hidden_size", default=512, type=int, nargs="+", help="dimensionality of hidden state in CWVAE")
    parser.add_argument("--latent_size", default=128, type=int, nargs="+", help="dimensionality of latent state in CWVAE")
    parser.add_argument("--time_factors", default=[200, 800, 3200], type=int, nargs="+", help="temporal abstraction factor")
    parser.add_argument("--num_level_layers", default=8, type=int, help="dense layers for embedding per level")
    parser.add_argument("--input_coding", default="mu_law", type=str, choices=["mu_law", "frames"], help="input encoding")
    parser.add_argument("--num_bits", default=8, type=int, help="number of bits for DML and input")
    parser.add_argument("--num_mix", default=10, type=int, help="number of logistic mixture components")
    parser.add_argument("--residual_posterior", default=False, type=str2bool, help="residual parameterization of posterior")
    parser.add_argument("--beta_anneal_steps", default=0, type=int, help="number of steps to anneal beta")
    parser.add_argument("--beta_start_value", default=0, type=float, help="initial beta annealing value")
    parser.add_argument("--free_nats_steps", default=0, type=int, help="number of steps to constant/anneal free bits")
    parser.add_argument("--free_nats_start_value", default=4, type=float, help="free bits per timestep")

    parser.add_argument("--epochs", default=750, type=int, help="number of epochs")

    parser.add_argument("--num_workers", default=4, type=int, help="number of dataloader workers")
    parser.add_argument("--seed", default=None, type=int, help="random seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    if args.seed is None:
        args.seed = get_random_seed()

    args.world_size = args.gpus * args.nodes

    os.environ["MASTER_ADDR"] = args.ddp_master_addr
    os.environ["MASTER_PORT"] = args.ddp_master_port

    mp.spawn(run, nprocs=args.gpus, args=(args,))


def run(gpu_idx, args):

    rank = args.node_rank * args.gpus + gpu_idx
    distributed.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=rank)
    torch.cuda.set_device(gpu_idx)

    set_seed(args.seed)

    if rank == 0:
        wandb.init(
            entity="vseq",
            project="cwvae",
            group=None,
        )
        wandb.config.update(args)
        rich.print(vars(args))


    # model = vseq.models.CWVAEAudioConv1d(
    #     z_size=args.latent_size,
    #     h_size=args.hidden_size,
    #     time_factors=args.time_factors,
    #     num_level_layers=args.num_level_layers,
    #     num_mix=args.num_mix,
    #     num_bins=2 ** args.num_bits,
    #     residual_posterior=args.residual_posterior
    # )
    model = vseq.models.CWVAEAudioTasNet(
        z_size=args.latent_size,
        h_size=args.hidden_size,
        time_factors=args.time_factors,
        num_level_layers=args.num_level_layers,
        num_mix=args.num_mix,
        num_bins=2 ** args.num_bits,
        residual_posterior=args.residual_posterior
    )
    # model = vseq.models.CWVAEAudioDense(
    # # model = vseq.models.CWVAEAudioConv1D(
    #     z_size=args.latent_size,
    #     h_size=args.hidden_size,
    #     time_factors=args.time_factors,
    #     num_level_layers=args.num_level_layers,
    #     num_mix=args.num_mix,
    #     num_bins=2 ** args.num_bits,
    #     residual_posterior=args.residual_posterior
    # )
    # model = vseq.models.CWVAEAudioCPCPretrained(
    #     z_size=args.latent_size,
    #     h_size=args.hidden_size,
    #     time_factors=args.time_factors,
    #     num_level_layers=args.num_level_layers,
    #     num_mix=args.num_mix,
    #     num_bins=2 ** args.num_bits,
    #     residual_posterior=args.residual_posterior
    # )


    decode_transform = []
    encode_transform = []
    if args.input_coding == "mu_law":
        encode_transform.append(MuLawEncode(bits=8))  #args.num_bits))
        decode_transform.append(MuLawDecode(bits=8))  #args.num_bits))

    # encode_transform.extend([Quantize(bits=8, rescale=True)])
    encode_transform = Compose(*encode_transform)
    decode_transform = Compose(*decode_transform)

    batcher = AudioBatcher(min_length=model.receptive_field, padding_module=model.overall_stride)
    # batcher = AudioBatcher(padding_module=model.overall_stride)
    loader = AudioLoader("wav", cache=False)
    modalities = [(loader, encode_transform, batcher)]

    train_dataset = BaseDataset(
        source=TIMIT_TRAIN,
        modalities=modalities,
    )
    valid_dataset = BaseDataset(
        source=TIMIT_TEST,
        modalities=modalities,
    )
    rich.print(train_dataset)

    train_sampler = DistributedSamplerWrapper(
        sampler=torch.utils.data.RandomSampler(train_dataset),
        num_replicas=args.world_size,
        rank=rank,
        drop_last=True,
    )
    valid_sampler = DistributedSamplerWrapper(
        sampler=torch.utils.data.BatchSampler(
            sampler=torch.utils.data.SequentialSampler(valid_dataset),
            batch_size=args.batch_size,
            drop_last=False,
        )
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=train_dataset.collate,
        num_workers=args.num_workers,
        sampler=train_sampler,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        collate_fn=valid_dataset.collate,
        num_workers=args.num_workers,
        sampler=valid_sampler,
        batch_size=args.batch_size,
        pin_memory=True,
    )


    if rank == 0:
        print(model)
        model.summary(input_size=(args.batch_size, model.overall_stride,), x_sl=torch.tensor([model.overall_stride]), device='cpu')
        wandb.watch(model, log="all", log_freq=len(train_loader))

    model = model.to(gpu_idx)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_idx])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    beta_annealer = CosineAnnealer(anneal_steps=args.beta_anneal_steps, start_value=args.beta_start_value, end_value=1)
    free_nats_annealer = CosineAnnealer(
        anneal_steps=args.free_nats_steps // 2,
        constant_steps=args.free_nats_steps // 2,
        start_value=args.free_nats_start_value,
        end_value=0,
    )

    tracker = Tracker(print_every=5.0, rank=rank, world_size=args.world_size)

    for epoch in tracker.epochs(args.epochs):

        model.train()
        for (x, x_sl), metadata in tracker(train_loader):
            x = x.to(gpu_idx, non_blocking=True)

            loss, metrics, outputs = model(x, x_sl, beta=beta_annealer.step(), free_nats=free_nats_annealer.step())

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            # torch.nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()

            tracker.update(metrics)

        model.eval()
        with torch.no_grad():
            for (x, x_sl), metadata in tracker(valid_loader):
                x = x.to(gpu_idx, non_blocking=True)

                loss, metrics, outputs = model(x, x_sl)

                tracker.update(metrics)


            extra = dict()
            if rank == 0 and epoch % 10 == 0:
                outputs.x_hat = decode_transform(outputs.x_hat)
                reconstructions = [wandb.Audio(outputs.x_hat[i].flatten().cpu().numpy(), caption=f"Reconstruction {i}", sample_rate=16000) for i in range(2)]

                (x, x_sl), outputs = model.generate(n_samples=2, max_timesteps=128000)
                x = decode_transform(x)
                samples = [wandb.Audio(x[i].flatten().cpu().numpy(), caption=f"Sample {i}", sample_rate=16000) for i in range(2)]

                # (x, x_sl), outputs = model.generate(n_samples=2, max_timesteps=128000, use_mode_observations=True)
                # x = decode_transform(x)
                # samples_mode = [wandb.Audio(x[i].flatten().cpu().numpy(), caption=f"Sample {i}", sample_rate=16000) for i in range(2)]

                # extra = dict(samples=samples, samples_mode=samples_mode, reconstructions=reconstructions)
                extra = dict(samples=samples, reconstructions=reconstructions)

            tracker.log()

    wandb.finish()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
