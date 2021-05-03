import csv
import os

from collections import Counter
from typing import Callable, Optional

from tqdm import tqdm

from vseq.settings import VOCAB_DIRECTORY

from .tokenizers import word_tokenizer
from .datapaths import DATAPATHS_MAPPING


def get_vocabulary_path(name: str):
    return os.path.join(VOCAB_DIRECTORY, f"{name}.txt")


def build_vocabulary(source: str, name: str, cleaner_fcn: Optional[Callable] = None):
    """
    Builds a vocabulary file with word-count pairs on each line.

    Args:
        source (str): A dataset name available in `DATAPATHS_MAPPING` or a path to a `source` file.
        name (str): A name for the vocabulary file.
        cleaner_fcn (Optional[Callable], optional): Callable to use for pre-cleaning the text data. Defaults to None.
    """
    source_filepath = DATAPATHS_MAPPING[source] if source in DATAPATHS_MAPPING else source
    with open(source_filepath, newline='') as source_file_buffer:
        reader = csv.DictReader(source_file_buffer)
        examples = [row["filename"] for row in reader]

    word_counts = Counter()
    for example in tqdm(examples, smoothing=0.05):
        file_path = example + ".txt"
        with open(file_path, "r") as text_file:
            strings = text_file.read().splitlines()

        for string in strings:
            clean_string = cleaner_fcn(string) if cleaner_fcn else string
            words = word_tokenizer(clean_string)
            word_counts.update(words)

    vocab_filepath = get_vocabulary_path(name)
    vocab_file_lines = [f"{w},{c}" for w, c in word_counts.most_common()]
    vocab_file_content = "\n".join(vocab_file_lines)
    with open(vocab_filepath, "w") as vocab_file_buffer:
        vocab_file_buffer.write(vocab_file_content)


def load_vocabulary(name, max_size=None, min_count=None):
    """
    Load vocabulary file corresponding to a given name.

    Args:
        name (str): Name of the vocabulary. Used to specify the vocabulary file.
        max_size (int): Maximum number of words to keep in the vocabulary.
        min_count (int): Minimum number of occurences of any word in vocabulary file.
    """
    vocab_filepath = get_vocabulary_path(name)

    max_size = float("inf") if max_size is None else max_size
    min_freq = 0 if min_count is None else min_count
    vocab = []
    with open(vocab_filepath, "r") as vocab_file_buffer:
        for line in vocab_file_buffer:
            word, count = line.strip().split(",")
            if int(count) < min_freq or len(vocab) >= max_size:
                break
            vocab.append(word)

    return vocab
