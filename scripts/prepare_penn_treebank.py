import os
import sys

import wget
from tqdm import tqdm

DATA_DESTINATION = "/data/research/" #sys.argv[1]
SOURCEFILE_DESTINATION = "/home/labo/repos/vseq/data/" #sys.argv[2]

SUBSETS = ["train", "test", "valid"]

assert os.path.isdir(DATA_DESTINATION), "Data destinations should be a directory."
assert os.path.isdir(SOURCEFILE_DESTINATION), "Source destinations should be a directory."

ptb_data_dir = os.path.join(DATA_DESTINATION, "penn_treebank")
ptb_sourcefile_dir = os.path.join(SOURCEFILE_DESTINATION, "penn_treebank")

assert not os.path.exists(ptb_data_dir), "Dataset already exists in data directory."
assert not os.path.exists(ptb_sourcefile_dir), "Dataset already exists in source directory."

os.mkdir(ptb_data_dir)
os.mkdir(ptb_sourcefile_dir)

for subset in SUBSETS:

    # download the subset
    download_url = f"https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{subset}.txt"
    print(f"\nDownloading - {subset}:")
    wget.download(download_url, ptb_data_dir)

    download_filepath = os.path.join(ptb_data_dir, f"ptb.{subset}.txt")
    with open(download_filepath, "r") as download_file_buffer:
        lines = download_file_buffer.readlines()
    
    subset_dir = os.path.join(ptb_data_dir, subset)
    os.mkdir(subset_dir)

    # create individual example files
    source_file_content = []
    for idx, line in tqdm(enumerate(lines)):
        example_id = (5 - len(str(idx))) * "0" + str(idx) 
        example_filename = f"{subset}_{example_id}.txt"
        example_file_path = os.path.join(subset_dir, example_filename)
        with open(example_file_path, "w") as example_file_buffer:
            example_file_buffer.write(line.strip())
        
        example_base_path = os.path.splitext(example_file_path)[0]
        source_file_content.append(example_base_path)

    # create the subset source file
    source_file_content = "\n".join(source_file_content)
    source_file_path = os.path.join(ptb_sourcefile_dir, f"{subset}.txt")
    with open(source_file_path, "w") as source_file_buffer:
        source_file_buffer.write(source_file_content)

    os.remove(download_filepath)
