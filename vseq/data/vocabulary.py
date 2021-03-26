import os
from collections import Counter

from tqdm import tqdm

from .tokenizers import word_tokenizer
from .datapaths import DATAPATHS_MAPPING

def _get_vocab_filepath(source_filepath, create_vocab_dir=False):
    """
    Forms, and possibly creates, the target path from the source file path as:
    {source_dir}/vocab/{source_filename}
    """
    source_dir, source_filename = os.path.split(source_filepath)
    vocab_dir = os.path.join(source_dir, "vocab")
    if not os.path.exists(vocab_dir) and create_vocab_dir:
        os.mkdir(target_dir)
    vocab_filepath = os.path.join(vocab_dir, source_filename)
    return vocab_filepath

def build_voabulary(source, cleaner_fcn=None):
    """
    Builds a vocabulary file with word-count pairs on each line.
    
    The file will use the same name as the source with '_vocab' suffix.

    A directorry 'vocab' is created at the source location.
    """
    cleaner_fcn = cleaner_fcn if cleaner_fcn is not None else lambda x: x

    source_filepath = DATAPATHS_MAPPING[source] if source in DATAPATHS_MAPPING else source
    with open(source_filepath, "r") as source_file_buffer:
        lines = source_file_buffer.readlines()
    examples = [l.split(',')[0] for l in lines]

    word_counts = Counter()
    for example in tqdm(examples):
        file_path = example + ".txt"
        with open(file_path, "r") as text_file:
            string = text_file.read()
        clean_string = cleaner_fcn(string)
        words = word_tokenizer(clean_string)
        word_counts.update(words)

    vocab_filepath = _get_vocab_filepath(source_filepath, create_vocab_dir=True)

    vocab_file_lines = [f"{w},{c}" for w, c in word_counts.most_common()]
    vocab_file_content = "\n".join(vocab_file_lines)
    with open(vocab_filepath, "w") as vocab_file_buffer:
        vocab_file_buffer.write(vocab_file_content)


def load_vocabualry(source, max_size=None, min_count=None):
    """
    Load vocabulary file corresponding to source.

    Args:
        source (str): A constant from vseq.data.datapaths or a path. Used to infer the vocab file.
        max_size (int): The 
    """
    source_filepath = DATAPATHS_MAPPING[source] if source in DATAPATHS_MAPPING else source
    vocab_filepath = _get_vocab_filepath(source_filepath, create_vocab_dir=True)

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