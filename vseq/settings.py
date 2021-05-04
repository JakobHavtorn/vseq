"""Settings file for vseq package"""

import os
import logging

import rootpath


ROOT_PATH = __file__.removesuffix("/vseq/settings.py")
ENV_FILE = os.path.join(ROOT_PATH, "VSEQ.env")

def read_env_file():
    if not os.path.exists(ENV_FILE):
        return dict()

    with open(ENV_FILE, "r") as envfile_buffer:
        env_text = envfile_buffer.read()
    return dict([line.split("=") for line in env_text.splitlines()])

ENV = read_env_file()

def set_vseq_envvar(envvar):
    envvar_value = input(f"\nRequired environment variable {envvar} not set.\n\nPlease specify:\n> ")
    with open(ENV_FILE, "a") as envfile_buffer:
        envfile_buffer.write(f"{envvar}={envvar_value}\n")
    return envvar_value

def get_vseq_envvar(envvar, default=None):
    
    if envvar in os.environ:
        return os.getenv(envvar)
    if envvar in ENV:
        return ENV[envvar]
    if default is None:
        return set_vseq_envvar(envvar)
    return default

# Logging
LOG_FORMAT = "%(asctime)-15s - %(module)-20s - %(levelname)-7s | %(message)s"
LOG_LEVEL = get_vseq_envvar("VSEQ_LOG_LEVEL", "INFO")
logging.basicConfig(format=LOG_FORMAT, level=logging.getLevelName(LOG_LEVEL))

DATA_ROOT_DIRECTORY = get_vseq_envvar("VSEQ_DATA_ROOT_DIRECTORY", default=None)
DATA_DIRECTORY = os.path.join(DATA_ROOT_DIRECTORY, "data")
SOURCE_DIRECTORY = os.path.join(DATA_ROOT_DIRECTORY, "source")
VOCAB_DIRECTORY = os.path.join(DATA_ROOT_DIRECTORY, "vocabularies")


for path in [DATA_DIRECTORY, SOURCE_DIRECTORY, VOCAB_DIRECTORY]:
    os.makedirs(path, exist_ok=True)
