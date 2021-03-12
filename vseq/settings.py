"""Settings file for vseq package"""

import os
import logging

import rootpath


# Logging
LOG_FORMAT = "%(asctime)-15s - %(module)-20s - %(levelname)-7s | %(message)s"
LOG_LEVEL = os.getenv("VSEQ_LOG_LEVEL", "INFO")

logging.basicConfig(format=LOG_FORMAT, level=logging.getLevelName(LOG_LEVEL))

# Turn down log level of Trains
logging.getLogger(name="trains.Task").setLevel(logging.WARNING)


# Log all environment variables to Trains by default
os.environ["TRAINS_LOG_ENVIRONMENT"] = os.getenv("TRAINS_LOG_ENVIRONMENT", "*")


# Directories
ROOT_PATH = rootpath.detect()
NAS_PATH = os.path.join("nas", "experiments", "research")

TENSORBOARD_DIRECTORY = os.getenv("VSEQ_TENSORBOARD_DIRECTORY", os.path.join(NAS_PATH, "tensorboard"))
TRAINS_DIRECTORY = os.getenv("VSEQ_TRAINS_DIRECTORY", os.path.join(NAS_PATH, "trains"))
DATA_DIRECTORY = os.getenv("VSEQ_DATA_DIRECTORY", os.path.join(ROOT_PATH, "data"))

TRAINS_DIRECTORY_NAS = os.getenv("VSEQ_TRAINS_DIRECTORY_NAS", os.path.join("/nas/experiments/trains"))


for path in [TRAINS_DIRECTORY, DATA_DIRECTORY]:
    os.makedirs(path, exist_ok=True)
