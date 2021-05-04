import os
import wget

from vseq.settings import DATA_DIRECTORY, SOURCE_DIRECTORY


SUBSETS = ["train", "test", "valid"]

ptb_data_dir = os.path.join(DATA_DIRECTORY, "penn_treebank")
ptb_source_dir = os.path.join(SOURCE_DIRECTORY, "penn_treebank")

assert not os.path.exists(ptb_data_dir), "Dataset already exists in data directory."
assert not os.path.exists(ptb_source_dir), "Dataset already exists in source directory."

os.mkdir(ptb_data_dir)
os.mkdir(ptb_source_dir)

header = "filename,n_examples"
for subset in SUBSETS:

    # download the subset
    download_url = f"https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{subset}.txt"
    print(f"\nDownloading - {subset}:")
    wget.download(download_url, ptb_data_dir)
    
    # count examples
    data_file_path = os.path.join(ptb_data_dir, f"ptb.{subset}.txt")
    with open(data_file_path, "r") as data_file_buffer:
        n_examples = data_file_buffer.read().strip().count("\n") + 1

    # create the subset source file
    source_file_lines = [header, f"{os.path.splitext(data_file_path)[0]},{n_examples}"]
    source_file_content = "\n".join(source_file_lines)
    source_file_path = os.path.join(ptb_source_dir, f"{subset}.txt")
    with open(source_file_path, "w") as source_file_buffer:
        source_file_buffer.write(source_file_content)

print("\n\nPenn Treebank dataset succesfully processed!")
