import os
import tarfile
import shutil
import wget

from glob import glob
from pathlib import Path
from tqdm import tqdm

import torchaudio

from vseq.settings import DATA_DIRECTORY, SOURCE_DIRECTORY


SUBSETS = ["dev-clean",
           "dev-other",
           "test-clean",
           "test-other",
           "train-clean-100",
           "train-clean-360",
           "train-other-500"]

librispeech_data_dir = os.path.join(DATA_DIRECTORY, 'librispeech')
librispeech_source_dir = os.path.join(SOURCE_DIRECTORY, 'librispeech')

assert not os.path.exists(librispeech_data_dir), "Dataset already exists in data directory."
assert not os.path.exists(librispeech_source_dir), "Dataset already exists in source directory."

os.mkdir(librispeech_data_dir)
os.mkdir(librispeech_source_dir)

header = ["filename,length.flac.samples\n"]
train_source_file_content = []
metadata_copied = False
for subset in SUBSETS:

    # download the subset
    download_url = f"https://www.openslr.org/resources/12/{subset}.tar.gz"
    print(f"\nDownloading - {subset}:")
    wget.download(download_url, librispeech_data_dir)
    
    # unzip download
    download_filepath = os.path.join(librispeech_data_dir, f"{subset}.tar.gz")
    tar = tarfile.open(download_filepath, "r:gz")
    tar.extractall(path=librispeech_data_dir)
    tar.close()

    # discard top-level folder and remove duplicated metadata
    download_dir = os.path.join(librispeech_data_dir, "LibriSpeech")
    subset_dir = os.path.join(download_dir, subset)
    new_subset_dir = os.path.join(librispeech_data_dir, subset)
    Path(subset_dir).rename(new_subset_dir)

    # move metadata files if not already done
    if not metadata_copied:    
        for metadata_filepath in glob(f"{download_dir}/*.TXT"):
            new_metadata_filepath = metadata_filepath.replace("LibriSpeech/", "")
            Path(metadata_filepath).rename(new_metadata_filepath)
        metadata_copied = True
    
    # clean-up unused files and folders
    shutil.rmtree(download_dir)
    os.remove(download_filepath)

    # split transcript files into single utterances
    print(f"\n\nSplitting transcript files - {subset}:")
    transcript_filepaths = glob(os.path.join(new_subset_dir, "*/*/*.trans.txt"))
    source_file_content = []
    for transcript_filepath in tqdm(transcript_filepaths):
        with open(transcript_filepath, "r") as transcript_file_buffer:
            lines = transcript_file_buffer.readlines()
        
        transcript_dir = os.path.split(transcript_filepath)[0]
        for line in lines:
            line = line.split()
            basename = line[0]
            transcript = " ".join(line[1:])
            new_transcript_filepath = os.path.join(transcript_dir, f"{basename}.txt")
            with open(new_transcript_filepath, "w") as new_transcript_file_buffer:
                new_transcript_file_buffer.write(transcript)
            
            audio_filepath = new_transcript_filepath.replace('txt', 'flac')
            audio_metadata = torchaudio.info(audio_filepath)
            source_file_line = os.path.join(transcript_dir, basename)
            source_file_content.append(f"{source_file_line},{audio_metadata.num_frames}\n")
        
        os.remove(transcript_filepath)
    
    # create the subset source file
    if subset.startswith("train"):
        train_source_file_content.extend(source_file_content) # makes a copy, so ok to modify below
    source_filepath = os.path.join(librispeech_source_dir, f"{subset}.txt")
    source_file_content[-1] = source_file_content[-1].strip()
    source_file_content = header + source_file_content
    with open(source_filepath, "w") as source_file_buffer:
        source_file_buffer.writelines(source_file_content)

# create full train source file
train_source_filepath = os.path.join(librispeech_source_dir, "train.txt")
train_source_file_content[-1] = train_source_file_content[-1].strip()
train_source_file_content = header + train_source_file_content
with open(train_source_filepath, "w") as train_source_file_buffer:
    train_source_file_buffer.writelines(train_source_file_content)

print("\nLibriSpeech dataset succesfully processed!")