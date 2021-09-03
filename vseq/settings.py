"""Settings file for vseq package"""

import os
import logging

from typing import Any


ROOT_PATH = __file__.replace("/vseq/settings.py", "")
ENV_FILE = os.path.join(ROOT_PATH, "VSEQ.env")


def read_env_file():
    if not os.path.exists(ENV_FILE):
        return dict()

    with open(ENV_FILE, "r") as envfile_buffer:
        env_text = envfile_buffer.read()
    return dict([line.split("=") for line in env_text.splitlines()])


ENV = read_env_file()


def write_vseq_envvar(envvar: str, envvar_value: str):
    """Write an envvar to the ENV_FILE"""
    with open(ENV_FILE, "a") as envfile_buffer:
        envfile_buffer.write(f"{envvar}={envvar_value}\n")


def require_vseq_envvar(envvar):
    """Request user for value of a required envvar and write it to the ENV_FILE"""
    envvar_value = input(f"\nRequired environment variable {envvar} not set.\n\nPlease specify:\n> ")
    write_vseq_envvar(envvar, envvar_value)
    return envvar_value


def get_vseq_envvar(envvar, default: Any = None, reflect: bool = False):
    """Retrieve the value of an envvar.

    In prioritized order, returns the value found in `os.environ`, `ENV_FILE`, `default`.

    If `default` is `None`, requires the envvar to be set and requests it from the user at runtime.
    If `reflect` is `True`, reflects the retrieved value into `os.environ`.
    """
    if envvar in os.environ:
        return os.getenv(envvar)

    if envvar in ENV:
        value = ENV[envvar]
    elif default is None:
        value = require_vseq_envvar(envvar)
    else:
        value = default

    if reflect:
        os.environ[envvar] = value

    return value


# logging
LOG_FORMAT = "%(asctime)-15s - %(module)-20s - %(levelname)-7s | %(message)s"
LOG_LEVEL = get_vseq_envvar("VSEQ_LOG_LEVEL", "WARNING")
logging.basicConfig(format=LOG_FORMAT, level=logging.getLevelName(LOG_LEVEL))

# data directories
DATA_ROOT_DIRECTORY = get_vseq_envvar("VSEQ_DATA_ROOT_DIRECTORY", default=None)
DATA_DIRECTORY = os.path.join(DATA_ROOT_DIRECTORY, "data")
SOURCE_DIRECTORY = os.path.join(DATA_ROOT_DIRECTORY, "source")
VOCAB_DIRECTORY = os.path.join(DATA_ROOT_DIRECTORY, "vocabularies")

# set wandb directory to inside data directory
WANDB_DIR = get_vseq_envvar("WANDB_DIR", default=DATA_ROOT_DIRECTORY, reflect=True)
CHECKPOINT_DIR = os.path.join(WANDB_DIR, "wandb")

# make directories
for path in [DATA_DIRECTORY, SOURCE_DIRECTORY, VOCAB_DIRECTORY, WANDB_DIR]:
    os.makedirs(path, exist_ok=True)
