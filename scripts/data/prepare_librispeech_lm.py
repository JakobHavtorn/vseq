import os
import gzip
import wget
import random

from math import ceil
from tqdm import tqdm

from vseq.settings import DATA_DIRECTORY, SOURCE_DIRECTORY


URL = "https://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz"

librilm_data_dir = os.path.join(DATA_DIRECTORY, 'librispeech_lm')
librilm_source_dir = os.path.join(SOURCE_DIRECTORY, 'librispeech_lm')

assert not os.path.exists(librilm_data_dir), "Dataset already exists in data directory." 
assert not os.path.exists(librilm_source_dir), "Dataset already exists in source directory."

os.mkdir(librilm_data_dir)
os.mkdir(librilm_source_dir)

# download, unzip and decode
print(f"\nDownloading:")
wget.download(URL, out=librilm_data_dir)
download_path = os.path.join(librilm_data_dir, "librispeech-lm-norm.txt.gz")

print(f"\n\nUnzipping and decoding - may take a minute...")
with gzip.open(download_path, "rb") as file_buffer:
    data = file_buffer.read().decode("utf-8")

# create 40 chunks from the full data file
num_data_files = 40
examples = data.strip().splitlines()
random.shuffle(examples)
max_examples_pr_file = ceil(len(examples) / num_data_files)

print("\nSaving data files...")
source_file_lines = ["filename,n_examples"]
for idx in tqdm(range(num_data_files)):
    data_file_path = os.path.join(librilm_data_dir, f"librispeech-lm-norm-{idx}.txt")
    start = idx * max_examples_pr_file
    stop = (idx + 1) * max_examples_pr_file
    data_chunk = examples[start:stop]
    source_file_line = f"{os.path.splitext(data_file_path)[0]},{len(data_chunk)}"
    source_file_lines.append(source_file_line)
    with open(data_file_path, "w") as data_file_buffer:
        data_file_buffer.write("\n".join(data_chunk))
    
os.remove(download_path)

# create the subset source file
source_file_content = "\n".join(source_file_lines)
source_file_path = os.path.join(librilm_source_dir, "librispeech_lm.txt")
with open(source_file_path, "w") as source_file_buffer:
    source_file_buffer.write(source_file_content)

print("\nLibriSpeech LM dataset succesfully processed!")