"""
This script prepares the TIMIT dataset.

Since this dataset is commercial, we do not download it in this script. That must be done beforehand.

This script then unpacks the downloaded data and creates a source file including file lengths (text and audio).
"""

import os
import sys

import torchaudio

from tqdm import tqdm
from glob import glob

from vseq.settings import DATA_DIRECTORY, SOURCE_DIRECTORY


SUBSETS = ["test", "train"]

data_dir = os.path.join(DATA_DIRECTORY, "timit")
source_dir = os.path.join(SOURCE_DIRECTORY, "timit")

assert os.path.exists(data_dir), "TIMIT dataset must already be downloaded."
assert not os.path.exists(source_dir), "Dataset already exists in source directory."

os.mkdir(source_dir)


header = "filename,length.wav.samples,length.txt.char,length.txt.word"
for subset in SUBSETS:

    # run check on files
    subset_data_dir = os.path.join(data_dir, subset)
    wav_filepaths = sorted(glob(os.path.join(subset_data_dir, "**/*.wav"), recursive=True))
    txt_filepaths = sorted(glob(os.path.join(subset_data_dir, "**/*.TXT"), recursive=True))
    assert len(wav_filepaths) == len(txt_filepaths)
    assert len(set(wav_filepaths)) == len(wav_filepaths)
    assert len(set(txt_filepaths)) == len(txt_filepaths)

    # load files and compute length
    extension_less_filepaths = [fp.replace(".wav", "") for fp in wav_filepaths]

    source_file_lines = []
    for file_path in tqdm(extension_less_filepaths):
        length_samples = torchaudio.info(file_path + ".wav").num_frames

        with open(file_path + ".TXT", "r") as data_file_buffer:
            txt = data_file_buffer.read()
            txt = txt.split()[2:]  # Remove pre annotation of ?? ('0 46797 She had your dark suit in greasy wash water all year.\n)

            length_char = len(" ".join(txt))
            length_word = len(txt)

        line = f"{file_path},{length_samples},{length_char},{length_word}"
        source_file_lines.append(line)

    # create the subset source file
    source_file_lines = [header] + source_file_lines
    source_file_content = "\n".join(source_file_lines)
    source_file_path = os.path.join(source_dir, f"{subset}.txt")
    with open(source_file_path, "w") as source_file_buffer:
        source_file_buffer.write(source_file_content)

    print(f"Saved source file at {source_file_path} of size {sys.getsizeof(source_file_content)} bytes")

print("\n\nTIMIT dataset succesfully processed!")
