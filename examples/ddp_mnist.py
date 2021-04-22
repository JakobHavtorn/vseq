"""
Basic example of Distributed Data Parallel (DDP) training a classifier on FashionMNIST.

Run with e.g.

> `env CUDA_VISIBLE_DEVICES=1,2 python examples/ddp_mnist.py --nodes 1 --gpus 2 --epochs 40`
"""

import argparse
import os

from datetime import datetime
from vseq.evaluation.tracker import Tracker
from vseq.evaluation.metrics import AccuracyMetric, LossMetric

import wandb
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nodes", default=1, type=int, help="number of nodes")
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("-nr", default=0, type=int, help="ranking of this node within the nodes")
    parser.add_argument("--epochs", default=2, type=int, help="number of total epochs to run")
    parser.add_argument("--num_workers", default=2, type=int, help="number of dataloading workers")
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    mp.spawn(train, nprocs=args.gpus, args=(args,))


class ConvNet(nn.Module):
    def __init__(self, conv1_ch=16, conv2_ch=16, hidden_dim=16, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, conv1_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(conv1_ch),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(conv1_ch, conv2_ch, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(conv2_ch),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * conv2_ch, hidden_dim),
            nn.ReLU(),
            nn.Batchidden_dimNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Batchidden_dimNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu_idx, args):
    rank = args.nr * args.gpus + gpu_idx
    dist.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=rank)

    torch.manual_seed(0)
    batch_size = 64

    if gpu_idx == 0:
        wandb.init(
            entity="vseq",
            project="sandbox",
            group=None,
        )

    # define the model
    model = ConvNet()
    if gpu_idx == 0:
        print(model)
    torch.cuda.set_device(gpu_idx)
    model.cuda(gpu_idx)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_idx])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu_idx)
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)

    # Data loading code
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    train_dataset.source = "mnist_train"
    valid_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )
    valid_dataset.source = "mnist_valid"
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=args.world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=valid_sampler,
    )

    tracker = Tracker(print_every=1.0, rank=rank, world_size=args.world_size)

    start = datetime.now()
    for epoch in tracker.epochs(args.epochs):

        model.train()
        for (images, labels) in tracker.steps(train_loader):
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
            for i, (images, labels) in enumerate(tracker(valid_loader)):
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

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
