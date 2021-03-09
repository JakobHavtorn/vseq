import argparse
import sys
import unittest.mock

import pytest

from infotropy.data import DataModule
from infotropy.constants import TRAIN_SPLIT, VAL_SPLIT


def test_init_default():
    dm = DataModule(train_datasets=["MNISTBinarized"], val_datasets=["MNISTBinarized", "FashionMNISTBinarized"])

    assert len(dm.train_datasets) == 1
    assert len(dm.val_datasets) == 2
    assert len(dm.test_datasets) == 0
    assert len(dm.train_dataset.datasets) == 1

    assert dm.train_datasets["MNISTBinarized"].split == TRAIN_SPLIT
    assert dm.val_datasets["MNISTBinarized"].split == VAL_SPLIT
    assert dm.val_datasets["FashionMNISTBinarized"].split == VAL_SPLIT


def test_init_dataset_configs():
    dm = DataModule(
        train_datasets={"MNISTBinarized": {"threshold": 0.6, "seed": 42}},
        val_datasets={"MNISTBinarized": {"threshold": 0.9}, "FashionMNISTBinarized": {"threshold": 0.3}},
        batch_size=64,
        data_workers=2,
    )

    assert dm.train_datasets["MNISTBinarized"].threshold == 0.6
    assert dm.train_datasets["MNISTBinarized"].seed == 42
    assert dm.val_datasets["MNISTBinarized"].threshold == 0.9
    assert dm.val_datasets["FashionMNISTBinarized"].threshold == 0.3

    assert dm.val_loaders["FashionMNISTBinarized"].batch_size == 64 * dm.test_batch_size_factor
    assert dm.val_loaders["FashionMNISTBinarized"].num_workers == 2

    assert dm.val_loaders["MNISTBinarized"].batch_size == 64 * dm.test_batch_size_factor
    assert dm.val_loaders["MNISTBinarized"].num_workers == 2


def test_init_cli_args():
    """Test dataset_configs argument takes precedence over CLI arguments for individual datasets"""
    # Minor pre-test that the sys.argv context manager works as expected
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_argument", type=int)

    test_args = ["--test_argument", "5"]
    with unittest.mock.patch("sys.argv", ["program_name", *test_args]):
        args, unknown = parser.parse_known_args()

    assert args.test_argument == 5

    # Test that CLI arguments are passed to all datasets that have that argument
    test_args = [
        "--threshold",
        "0.2",
    ]
    with unittest.mock.patch("sys.argv", ["program_name", *test_args]):
        dm = DataModule(
            train_datasets=["MNISTBinarized"],
            val_datasets=["MNISTBinarized", "FashionMNISTBinarized"],
        )
    assert dm.train_datasets["MNISTBinarized"].threshold == 0.2
    assert dm.val_datasets["MNISTBinarized"].threshold == 0.2
    assert dm.val_datasets["FashionMNISTBinarized"].threshold == 0.2

    # Test that CLI arguments are preceeded by kwargs
    with unittest.mock.patch("sys.argv", "program_name", *test_args):
        dm = DataModule(
            train_datasets={"MNISTBinarized": {"threshold": 0.6}},
            val_datasets=["MNISTBinarized"],
        )

    assert dm.train_datasets["MNISTBinarized"].threshold == 0.6
    assert dm.val_datasets["MNISTBinarized"].threshold == 0.5  # default (TODO .default_threshold)


def test_batch_size():
    dm = DataModule(train_datasets=["MNISTBinarized"], val_datasets=["MNISTBinarized"], batch_size=64)

    dm.batch_size = 256

    assert dm.train_loader.batch_size == 256
    assert dm.val_loaders["MNISTBinarized"].batch_size == 64 * dm.test_batch_size_factor

    dm.test_batch_size = 1024

    assert dm.train_loader.batch_size == 256
    assert dm.val_loaders["MNISTBinarized"].batch_size == 1024
