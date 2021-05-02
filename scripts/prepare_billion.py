import os
import shutil
import tarfile
from glob import glob

import wget
from tqdm import tqdm

from vseq.settings import DATA_DIRECTORY, SOURCE_DIRECTORY

SUBSETS = ["training", "heldout"]

URL = "https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"

billion_data_dir = os.path.join(DATA_DIRECTORY, 'billion')
billion_source_dir = os.path.join(SOURCE_DIRECTORY, 'billion')

assert not os.path.exists(billion_data_dir), "Dataset already exists in data directory." 
assert not os.path.exists(billion_source_dir), "Dataset already exists in source directory."

os.mkdir(billion_data_dir)
os.mkdir(billion_source_dir)

print(f"\nDownloading:")
wget.download(URL, out=billion_data_dir)

print(f"\n\nUnzipping and writing - may take a minute...")
download_file_path = os.path.join(billion_data_dir, "1-billion-word-language-modeling-benchmark-r13output.tar.gz")
tar = tarfile.open(download_file_path, "r:gz")
tar.extractall(path=billion_data_dir)
tar.close()

# move files and dirs to avoid redundant top-level dir
glob_exp = os.path.join(billion_data_dir, "1-billion-word-language-modeling-benchmark-r13output/*")
items_to_move = glob(glob_exp)
for current_path in items_to_move:
    new_path = current_path.replace("1-billion-word-language-modeling-benchmark-r13output/", "")
    shutil.move(current_path, new_path)

# clean up tar.gz, empty dir and duplicates
duplicate_file_path = os.path.join(billion_data_dir, "heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100")
os.remove(duplicate_file_path)
shutil.rmtree(download_file_path.removesuffix(".tar.gz"))
os.remove(download_file_path)

# create the source files
print("\nCounting examples and creating source files...")
for subset in SUBSETS:
    
    data_file_paths = glob(os.path.join(billion_data_dir, f"{subset}-monolingual.tokenized.shuffled/*"))
    data_file_paths = sorted(data_file_paths)

    print(f"\nProcessing {subset} subset:")
    source_file_lines = ["filename,n_examples"]
    for data_file_path in tqdm(data_file_paths):
        with open(data_file_path, "r") as data_file_buffer:
            n_examples = data_file_buffer.read().strip().count("\n") + 1
        source_file_lines.append(f"{data_file_path},{n_examples}")
        shutil.move(data_file_path, data_file_path + ".txt") # add txt extension
    
    source_file_content = "\n".join(source_file_lines)
    source_file_path = os.path.join(billion_source_dir, f"{subset}.txt")
    with open(source_file_path, "w") as source_file_buffer:
        source_file_buffer.write(source_file_content)

print("\n1 Billion Word Language Model Benchmark dataset succesfully processed!")
    