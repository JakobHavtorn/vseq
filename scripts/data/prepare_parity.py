"""Generates a synthetic dataset for the so-called Parity Task

Given a sequence of fixed length L with elements either 0, 1 or -1, predict the parity of the sequence.

The parity is defined as 1 if there is an odd number of ones (positive or negative) and 0 if there is an even number. 

The sequence elements are sampled completely randomly, i.e. 
- First a number of elements, M, to set to ones (positive or negative) is sampled uniformly between 1 and L.
- Second, M elements are sampled with replacement from {-1, 1}, i.e. sample random number of positive and negative ones.
"""

import os

import numpy as np

from vseq.settings import DATA_DIRECTORY, SOURCE_DIRECTORY

SEQ_LENGTH = 64
SEED = 0
NUM_TRAIN_EXAMPLES = 60000
NUM_VALID_EXAMPLES = 10000
NUM_TEST_EXAMPLES = 10000

NUM_EXAMPLES = NUM_TRAIN_EXAMPLES + NUM_VALID_EXAMPLES + NUM_TEST_EXAMPLES

SUBSETS = ["train", "test", "valid"]

data_dir = os.path.join(DATA_DIRECTORY, "parity")
source_dir = os.path.join(SOURCE_DIRECTORY, "parity")

assert not os.path.exists(data_dir), "Dataset already exists in data directory."
assert not os.path.exists(source_dir), "Dataset already exists in source directory."

os.mkdir(data_dir)
os.mkdir(source_dir)


import IPython; IPython.embed(using=False)

np.random.seed(SEED)

num_ones = np.random.randint(1, SEQ_LENGTH, NUM_EXAMPLES)

num_positive = []
for n_ones in num_ones:
    n_positive = np.random.randint(0, n_ones)
    num_positive.append(n_positive)

examples = []
for i in range(NUM_EXAMPLES):
    one_indices = np.random.randint(0, SEQ_LENGTH, num_ones[i])
    


header = "filename,n_examples"
for subset in SUBSETS:
    
    # count examples
    data_file_path = os.path.join(data_dir, f"ptb.{subset}.txt")
    with open(data_file_path, "r") as data_file_buffer:
        n_examples = data_file_buffer.read().strip().count("\n") + 1

    # create the subset source file
    source_file_lines = [header, f"{os.path.splitext(data_file_path)[0]},{n_examples}"]
    source_file_content = "\n".join(source_file_lines)
    source_file_path = os.path.join(source_dir, f"{subset}.txt")
    with open(source_file_path, "w") as source_file_buffer:
        source_file_buffer.write(source_file_content)

print("\n\nPenn Treebank dataset succesfully processed!")
