import os
import sys
import shutil

from numpy.lib.utils import source

from vseq.settings import DATA_DIRECTORY, SOURCE_DIRECTORY

dataset = sys.argv[1]

assert isinstance(dataset, str) and len(dataset) > 0

data_dir = os.path.join(DATA_DIRECTORY, dataset)
sourcefile_dir = os.path.join(SOURCE_DIRECTORY, dataset)

assert os.path.exists(data_dir), f"Dataset {dataset} does not exist at data directory {data_dir}."
shutil.rmtree(data_dir)

assert os.path.exists(sourcefile_dir), f"Dataset {dataset} does not exist at sourcefile directory {sourcefile_dir}."
shutil.rmtree(sourcefile_dir)
