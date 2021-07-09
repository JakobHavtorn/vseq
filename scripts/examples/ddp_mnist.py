"""
Basic example of Distributed Data Parallel (DDP) training a classifier on FashionMNIST.

Run with e.g.

> `env CUDA_VISIBLE_DEVICES=1,2 python examples/ddp_mnist.py --nodes 1 --gpus 2 --epochs 40`
"""

import argparse
import os

from datetime import datetime

import wandb
import torch.multiprocessing as mp
import torchinfo
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist

from vseq.data.samplers import DistributedSamplerWrapper
from vseq.evaluation.tracker import Tracker
from vseq.evaluation.metrics import AccuracyMetric, LossMetric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp_master_addr", default="localhost", type=str, help="address for the DDP master")
    parser.add_argument("--ddp_master_port", default="123456", type=str, help="port for the DDP master")
    parser.add_argument("-n", "--nodes", default=1, type=int, help="number of nodes")
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("-nr", "--node_rank", default=0, type=int, help="ranking of this node within the nodes")
    parser.add_argument("--lr", default=3e-4, type=float, help="batch size")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--epochs", default=100, type=int, help="number of total epochs to run")
    parser.add_argument("--num_workers", default=1, type=int, help="number of dataloading workers")
    parser.add_argument("--seed", default=0, type=int, help="seed")
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes

    os.environ["MASTER_ADDR"] = args.ddp_master_addr
    os.environ["MASTER_PORT"] = args.ddp_master_port

    mp.spawn(train, nprocs=args.gpus, args=(args,))


class ConvNet(nn.Module):
    def __init__(self, conv1_ch=64, conv2_ch=128, conv3_ch=256, conv4_ch=64, hidden_dim=512, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, conv1_ch, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(conv1_ch),
            nn.GroupNorm(num_groups=conv1_ch, num_channels=conv1_ch),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # (14, 14)
        self.layer2 = nn.Sequential(
            nn.Conv2d(conv1_ch, conv2_ch, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(num_groups=conv2_ch, num_channels=conv2_ch),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # (7, 7)
        self.layer3 = nn.Sequential(
            nn.Conv2d(conv2_ch, conv3_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=conv3_ch, num_channels=conv3_ch),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )  # (3, 3)
        self.fc = nn.Sequential(
            nn.Linear(3*3*conv3_ch, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu_idx, args):
    rank = args.node_rank * args.gpus + gpu_idx
    dist.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=rank)
    torch.cuda.set_device(gpu_idx)

    torch.manual_seed(args.seed)

    if rank == 0:
        wandb.init(
            entity="vseq",
            project="sandbox",
            group=None,
        )

    # define the model
    model = ConvNet()
    if rank == 0:
        print(model)
        torchinfo.summary(model, input_size=(16, 1, 28, 28), col_names=("input_size", "output_size", "num_params", "kernel_size", "mult_adds"))
    model.cuda(gpu_idx)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_idx])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu_idx)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # Data loading code
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=rank==0
    )
    train_dataset.source = 'train'
    valid_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=rank==0
    )
    valid_dataset.source = 'val'

    train_sampler = DistributedSamplerWrapper(
        sampler=torch.utils.data.RandomSampler(train_dataset),
        num_replicas=args.world_size,
        rank=rank
    )
    valid_sampler = DistributedSamplerWrapper(
        sampler=torch.utils.data.BatchSampler(
            sampler=torch.utils.data.SequentialSampler(valid_dataset),
            batch_size=args.batch_size,
            drop_last=False,
        )
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_sampler=valid_sampler
    )

    tracker = Tracker(print_every=1.0, rank=rank, world_size=args.world_size)

    start = datetime.now()
    for epoch in tracker.epochs(args.epochs):

        model.train()
        for images, labels in tracker.steps(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = [
                LossMetric(loss, weight_by=labels.numel()),
                AccuracyMetric(predictions=torch.max(outputs.data, 1).indices, labels=labels),
            ]

            tracker.update(metrics)

        with torch.no_grad():
            model.eval()
            for images, labels in tracker.steps(valid_loader):
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                metrics = [
                    LossMetric(loss, weight_by=labels.numel()),
                    AccuracyMetric(predictions=torch.max(outputs.data, 1).indices, labels=labels),
                ]

                tracker.update(metrics)

        tracker.log()

    if gpu_idx == 0:
        print("Training complete in: " + str(datetime.now() - start))

    wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
