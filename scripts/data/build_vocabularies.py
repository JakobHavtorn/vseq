import argparse
import logging
import os

from vseq.data.vocabulary import build_vocabulary, get_vocabulary_path
from vseq.data.datapaths import (
    DATAPATHS_MAPPING,
    LIBRISPEECH_TRAIN,
    PENN_TREEBANK_TRAIN,
    BILLION_TRAINING,
    LIBRISPEECH_LM,
)


LOGGER = logging.getLogger(name=__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", type=str, nargs="+", default=["all"])
args = parser.parse_args()


# argument validation
assert len(args.datasets) == len(set(args.datasets)), "Duplicate dataset entries found."
args.datasets = set(args.datasets)


# build vocabularies for all datasets if specified
if "all" in args.datasets:
    args.datasets.remove("all")
    args.datasets = args.datasets.union([LIBRISPEECH_TRAIN, PENN_TREEBANK_TRAIN, BILLION_TRAINING, LIBRISPEECH_LM])


# check for existing vocabulary
vocabulary_exists = {dset for dset in args.datasets if os.path.exists(get_vocabulary_path(dset))}
vocabulary_not_exists = args.datasets - vocabulary_exists
if vocabulary_exists:
    LOGGER.warning(f"Skipping the following datasets since their vocabularies are already built: {vocabulary_exists}")


# check for dataset being downloaded
datapaths = {dset: DATAPATHS_MAPPING[dset] for dset in args.datasets}
datasets_downloaded = {dset for dset, path in datapaths.items() if os.path.exists(path)}
datasets_not_downloaded = args.datasets - datasets_downloaded
if datasets_not_downloaded:
    LOGGER.warning(f"Skipping the following datasets since they are not downloaded: {datasets_not_downloaded}")


# build vocabularies
for dset in datasets_downloaded:
    LOGGER.info(f"Building vocabulary for {dset} at {get_vocabulary_path(dset)}")
    build_vocabulary(source=dset, name=dset)
