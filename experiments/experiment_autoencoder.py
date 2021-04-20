from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchmetrics
from torchvision.datasets.mnist import FashionMNIST
from torchvision import transforms

from pytorch_lightning.loggers import WandbLogger

from vseq.utils.rand import get_random_seed


class Loss(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("accumulated", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("observations", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor):
        loss = loss.detach()
        self.accumulated += loss.sum()
        self.observations += loss.numel()

    def compute(self):
        return self.accumulated / self.observations


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, hidden_dim: int = 64, latent_dim: int = 3, learning_rate: int = 3e-4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28)
        )

        self.train_metrics = torchmetrics.MetricCollection(dict(loss=Loss()))
        self.valid_metrics = torchmetrics.MetricCollection(dict(loss=Loss()))
        self.test_metrics = torchmetrics.MetricCollection(dict(loss=Loss()))

        # import IPython; IPython.embed()
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.infer(x)

    def training_step(self, batch, batch_idx):
        loss = self.full_forward(batch)
        self.log('train_step', self.train_metrics(loss=loss))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.full_forward(batch)
        self.log('valid_step', self.valid_metrics(loss=loss), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.full_forward(batch)
        self.log('test_step', self.test_metrics(loss=loss))

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_epoch', self.train_metrics.compute())
        self.log('valid_epoch', self.valid_metrics.compute())

    def test_epoch_end(self, outs):
        # log epoch metric
        self.log('test_epoch', self.test_metrics.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--latent_dim', type=int, default=32)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

    def full_forward(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.infer(x)
        x_hat = self.reconstruct(z)
        loss = self.compute_loss(x_hat, x)
        return loss

    def infer(self, x):
        embedding = self.encoder(x)
        return embedding

    def reconstruct(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def compute_loss(self, x_hat, x):
        return F.mse_loss(x_hat, x)


def cli_main():
    # ------------
    # args
    # ------------
    # import IPython; IPython.embed()
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser = LitAutoEncoder.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.seed is None:
        args.seed = get_random_seed()
    pl.seed_everything(args.seed)

    # ------------
    # logger
    # ------------
    logger = WandbLogger(
        entity="vseq",
        project="autoencoder",
        group=None,
    )

    # ------------
    # data
    # ------------
    dataset = FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # ------------
    # model
    # ------------
    model = LitAutoEncoder(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim, learning_rate=args.learning_rate)

    logger.watch(model, log='all', log_freq=len(train_loader))

    # ------------
    # training
    # ------------
    args.logger = logger
    args.auto_lr_find = True
    args.auto_select_gpus = True
    trainer = pl.Trainer.from_argparse_args(args)

    # call tune to find the lr
    # trainer.tune(model)

    # fit
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()
