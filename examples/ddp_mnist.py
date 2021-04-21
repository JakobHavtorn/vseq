import argparse
import time
import os

from datetime import datetime

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
    os.environ["MASTER_PORT"] = "12355"

    mp.spawn(train, nprocs=args.gpus, args=(args,))


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

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
    valid_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )
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

    start = datetime.now()
    total_train_step = len(train_loader)
    total_valid_step = len(valid_loader)
    for epoch in range(args.epochs):

        model.train()
        running_loss = 0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            accuracy = correct / total

            if gpu_idx == 0 and ((i + 1) % 10 == 0 or (i + 1) == total_train_step):
                print(
                    "Train Epoch [{:3d}/{:3d}], Step [{:3d}/{:3d}], Loss: {:.4f}, Acc: {:.4f}".format(
                        epoch + 1, args.epochs, i + 1, total_train_step, running_loss / (i + 1), accuracy,
                    ),
                end="\r")

        if gpu_idx == 0:
            print()
        else:
            time.sleep(1)
        print(f"{rank}: {loss=} {running_loss=} {accuracy=}")

        with torch.no_grad():
            model.eval()
            running_loss = 0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(valid_loader):
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                accuracy = correct / total

                if gpu_idx == 0 and ((i + 1) % 10 == 0 or (i + 1) == total_valid_step):
                    print(
                        "Valid Epoch [{:3d}/{:3d}], Step [{:3d}/{:3d}], Loss: {:.4f}, Acc: {:.4f}".format(
                            epoch + 1, args.epochs, i + 1, total_valid_step, running_loss / (i + 1), accuracy,
                        ),
                    end="\r")

        if gpu_idx == 0:
            print()
            print("----------------------------------------------------------------")


    if gpu_idx == 0:
        print("Training complete in: " + str(datetime.now() - start))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
