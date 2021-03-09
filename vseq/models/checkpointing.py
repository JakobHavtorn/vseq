import datetime
import logging
import os

from typing import List
from dataclasses import dataclass
from pathlib import Path, PosixPath

import torch

from tqdm import tqdm
from clearml import Task
from clearml_agent import APIClient

import infotropy.settings
import infotropy.utils

from infotropy.datasets import DataModule
from .base_module import load_model, BaseModule


LOGGER = logging.getLogger(name=__file__)


def get_task_paths_from_ids(uuids, source_directory=infotropy.settings.TRAINS_DIRECTORY):
    if isinstance(uuids, str):
        uuids = [uuids]
    paths = [path for uuid in uuids for path in Path(source_directory).rglob("*") if uuid in path.name]
    assert len(paths) == len(uuids), f"Somehow there where two matches for at least one uuid"
    return paths


def task_to_checkpoint_path(task_paths):
    if isinstance(task_paths, str):
        task_paths = [task_paths]
    paths = [path.joinpath("models") for path in task_paths]
    return paths


@dataclass
class Checkpoint:
    """Class that holds attributes of a checkpoint.
    
    Instantiate with `uuid` and `populate()` the remaining fields and use `load()` to load model
    """
    uuid: str
    path: PosixPath = None
    name: str = ""
    task: Task = None
    model: BaseModule = None
    optimizer: torch.optim.Optimizer = None
    datamodule: DataModule = None
    last_modified: datetime.datetime = None

    client = APIClient()

    def populate(self):
        if not self.task:
            self.task = self.client.tasks.get_by_id(self.uuid)
        if not self.path:
            self.path = task_to_checkpoint_path(get_task_paths_from_ids(self.uuid))[0]
        if not self.name:
            self.name = self.task.name  # self.get_experiment_name()
        if not self.last_modified:
            mod_s_since_epoch = os.path.getmtime(self.path)
            self.last_modified = datetime.datetime.fromtimestamp(mod_s_since_epoch).strftime('%Y-%m-%d %H:%M:%S')
        return self

    def load(self, device=infotropy.utils.get_device()):
        self.load_model(device=device)
        self.load_datamodule()
        return self

    def load_model(self, device=infotropy.utils.get_device()):
        self.model = infotropy.models.load_model(self.path, device=device)
        return self

    def load_datamodule(self, **override_kwargs):
        LOGGER.info('Loading DataModule' + f' with overridding kwargs {override_kwargs}' if override_kwargs else '')
        self.datamodule = DataModule.load(self.path, **override_kwargs)
        return self


@dataclass
class CheckpointList:
    """Class that holds a list of Checkpoints"""
    checkpoints: List[Checkpoint]

    def __getitem__(self, idx):
        return self.checkpoints[idx]

    def get_from_uuid(self, uuid):
        checkpoint = [checkpoint for checkpoint in self.checkpoints if checkpoint.uuid == uuid]
        assert len(checkpoint) <= 1, "Several checkpoints with this UUID!"
        if len(checkpoint) == 1:
            return checkpoint[0]
        return None

    def populate(self):
        for checkpoint in self.checkpoints:
            checkpoint.populate()

    def load(self, device=infotropy.utils.get_device()):
        for checkpoint in tqdm(self.checkpoints):
            checkpoint.load(device=device)

    def load_model(self, device=infotropy.utils.get_device()):
        for checkpoint in tqdm(self.checkpoints):
            checkpoint.load_model(device=device)

    def load_datamodule(self, **override_kwargs):
        for checkpoint in tqdm(self.checkpoints):
            checkpoint.load_datamodule(**override_kwargs)

    def __len__(self):
        return len(self.checkpoints)
