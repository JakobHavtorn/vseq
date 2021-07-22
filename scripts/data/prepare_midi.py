"""
Download and prepare the MIDI datasets from http://www-etud.iro.umontreal.ca/~boulanni/icml2012

The downloaded pickle files are dictionaries with 'train', 'valid' and 'test' keys, with the
corresponding values being a list of example sequences.
Each sequence is itself a list of time steps, and each time step is a list of the non-zero
elements in the piano-roll at this instant (in MIDI note numbers, between 21 and 108 inclusive)

The files we download here are piano-rolls generated from the source files by transposing each sequence
in C major or C minor and sampling frames every eighth note (quarter note for JSB chorales) following the
beat information present in the MIDI file.


> From "Modeling Temporal Dependencies in High-Dimensional Sequences", 2012 (https://arxiv.org/abs/1206.6392):

 - Piano-midi.de is a classical piano MIDI archive that was split according to Poliner & Ellis (2007).
 - Nottingham is a collection of 1200 folk tunes 3 with chords instantiated from the ABC format.
 - MuseData is an electronic library of orchestral and piano classical music from CCARH 4 .
 - JSB chorales refers to the entire corpus of 382 fourpart harmonized chorales by J. S. Bach with the split of Allan & Williams (2005).

Each dataset contains at least 7 hours of polyphonic music and the total duration is approximately 67 hours.
The polyphony (number of simultaneous notes) varies from 0 to 15 and the average polyphony is 3.9.
We use an input of 88 binary visible units that span the whole range of piano from A0 to C8 and temporally aligned
on an integer fraction of the beat (quarter note).
Consequently, pieces with diﬀerent time signatures will not have their measures start at the same interval.
Although it is not strictly necessary, learning is facilitated if the sequences are transposed in a common tonality
(e.g. C major/minor) as preprocessing.
"""


import os
import wget
import pickle

from typing import List

import numpy as np

from tqdm import tqdm

from vseq.settings import DATA_DIRECTORY, SOURCE_DIRECTORY


SOURCE_URL = "http://www-etud.iro.umontreal.ca/~boulanni"

DATASETS = [
    ("Piano-midi.de.pickle", "piano"),
    ("Nottingham.pickle", "nottingham"),
    ("MuseData.pickle", "muse_data"),
    ("JSB Chorales.pickle", "jsb_chorales"),
]

SUBSETS = ["train", "test", "valid"]

MAX_TONE = 108
MIN_TONE = 21
N_TONES = MAX_TONE - MIN_TONE + 1  # Both inclusive, =88


def binary_vector(vector: List[int], num_classes: int = N_TONES, offset: int = MIN_TONE):
    """Convert a list of integers to a binary vector"""
    vector = (np.array(vector) - offset).astype(np.int32)
    return np.eye(num_classes)[vector].sum(0)


midi_data_dir = os.path.join(DATA_DIRECTORY, "midi")
midi_source_dir = os.path.join(SOURCE_DIRECTORY, "midi")

assert not os.path.exists(midi_data_dir), "Dataset already exists in data directory."
assert not os.path.exists(midi_source_dir), "Dataset already exists in source directory."

os.mkdir(midi_data_dir)
os.mkdir(midi_source_dir)

for dataset_pickle, dataset_name in DATASETS:
    # download the dataset_pickle
    download_url = f"{SOURCE_URL}/{dataset_pickle}"
    data_file_path = os.path.join(midi_data_dir, dataset_pickle)
    print(f"\nDownloading - {data_file_path}:")
    wget.download(download_url, midi_data_dir)

    # List[List[List[int]]]
    # example - timestep - nonzero piano roll elements in [21, 108]
    print(f"\nProcessing - {dataset_pickle}:")
    with open(data_file_path, "rb") as data_file_buffer:
        contents = pickle.load(data_file_buffer)
    assert isinstance(contents, dict) and set(contents.keys()) == set(SUBSETS)

    # convert to List[np.ndarray] where each array is of shape [timesteps, N_TONES=88]
    for subset in SUBSETS:
        data = contents[subset]

        midi_data_subdir = os.path.join(midi_data_dir, dataset_name, subset)
        os.makedirs(midi_data_subdir, exist_ok=True)

        # convert
        new_data = [np.array([binary_vector(example_t) for example_t in example]).astype(np.float32) for example in data]
        n_examples = len(new_data)
        lengths = [example.shape[0] for example in new_data]

        # save as files
        filepaths = []
        for i, ndat in enumerate(new_data):
            filename = f"example_{str(i).zfill(len(str(n_examples)))}.npy"
            filepath = os.path.join(midi_data_subdir, filename)

            with open(filepath, "wb") as f:
                np.save(f, ndat)

            filepaths.append(filepath)

        # create the subset source file
        midi_source_subdir = os.path.join(midi_source_dir, dataset_name)
        source_file_path = os.path.join(midi_source_subdir, f"{subset}.txt")
        os.makedirs(midi_source_subdir, exist_ok=True)

        header = "filename,length.npy"
        source_file_lines = [header]
        source_file_lines.extend([f"{os.path.splitext(filepaths[i])[0]}, {lengths[i]}" for i in range(len(filepaths))])
        source_file_content = "\n".join(source_file_lines)

        with open(source_file_path, "w") as source_file_buffer:
            source_file_buffer.write(source_file_content)
        print(f"Created source file - {source_file_path}")
        
print("\n\nMIDI datasets succesfully processed!")
