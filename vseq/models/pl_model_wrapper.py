from argparse import ArgumentParser

import pytorch_lightning as pl

from torch.optim.optimizer import Optimizer

from vseq.models import BaseModel


class PLModelWrapper(pl.LightningModule):
    def __init__(self, model: BaseModel, optimizer: Optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.save_hyperparameters(model.init_arguments())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, metrics, outputs = self.model.forward(batch)
        self.log('train_step', {m.name: m.value for m in metrics})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics, outputs = self.model.forward(batch)
        self.log('valid_step', {m.name: m.value for m in metrics}, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics, outputs = self.model.forward(batch)
        self.log('test_step', {m.name: m.value for m in metrics})
        return loss

    def configure_optimizers(self):
        return self.optimizer
