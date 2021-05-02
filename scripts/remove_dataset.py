import os
import sys
import shutil

from vseq.settings import DATA_DIRECTORY, SOURCE_DIRECTORY

dataset = sys.argv[1]

assert isinstance(dataset, str) and len(dataset) > 0

data_dir = os.path.join(DATA_DIRECTORY, dataset)
sourcefile_dir = os.path.join(SOURCE_DIRECTORY, dataset)

assert os.path.exists(data_dir), "Dataset does not exist in data directory."
assert os.path.exists(sourcefile_dir), "Dataset does not exist in sourcefile directory."

shutil.rmtree(data_dir)
shutil.rmtree(sourcefile_dir)