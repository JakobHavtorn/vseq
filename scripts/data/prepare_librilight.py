import os
import tarfile
import shutil
from glob import glob

import wget

from vseq.data.datapaths import LIBRISPEECH_TRAIN, DATAPATHS_MAPPING
from vseq.settings import DATA_DIRECTORY, SOURCE_DIRECTORY

subsets = {"train-10h": [],
           "train-1h": [],
           "train-10m-0": [],
           "train-10m-1": [],
           "train-10m-2": [],
           "train-10m-3": [],
           "train-10m-4": [],
           "train-10m-5": []}



librilight_source_dir = os.path.join(SOURCE_DIRECTORY, 'librilight')

assert not os.path.exists(librilight_source_dir), "Dataset already exists in source directory."

os.mkdir(librilight_source_dir)

# download the subset
download_url = f"https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz"
print(f"\nDownloading LibriLight finetuning split:")
wget.download(download_url, librilight_source_dir)
print("\n\nSuccess!\n")
    
# unzip download
download_filepath = os.path.join(librilight_source_dir, "librispeech_finetuning.tgz")
tar = tarfile.open(download_filepath, "r:gz")
tar.extractall(path=librilight_source_dir)
tar.close()

# extract files for each subset non-overlapping subset
subset_subpaths = ["9h"] + [f"1h/{i}" for i in range(6)]
subsets_no = {}
for ss in subset_subpaths:
    pattern = os.path.join(librilight_source_dir, f"librispeech_finetuning/{ss}/*/*/*/*.flac")
    flac_paths = glob(pattern)
    flac_basenames = [os.path.basename(p).replace(".flac", "") for p in flac_paths]
    subsets_no[ss] = flac_basenames

# define overlapping subsets
for i in range(5):
    subsets[f"train-10m-{i}"] += subsets_no[f"1h/{i}"]
    subsets["train-1h"] += subsets_no[f"1h/{i}"]
    subsets["train-10h"] += subsets_no[f"1h/{i}"] 
subsets["train-10h"] += subsets_no["9h"]

# load librispeech mapping
train_source = DATAPATHS_MAPPING[LIBRISPEECH_TRAIN]
with open(train_source, "r") as train_file_buffer:
    lines = train_file_buffer.readlines()
header = lines[0]
lines = lines[1:]
mapping = {os.path.basename(l.split(",")[0]) : l for l in lines}

# construct source files
for subset_name, examples in subsets.items():
    source_file = [header]
    for basename in examples:
       source_file.append(mapping[basename])
    source_file = "".join(source_file).strip()
    subset_source_path = os.path.join(librilight_source_dir, f"{subset_name}.txt")
    with open(subset_source_path, "w") as source_file_buffer:
        source_file_buffer.write(source_file)

os.remove(download_filepath)
shutil.rmtree(os.path.splitext(download_filepath)[0])
print("\nSource files created!")





