import time
import timeit

from numpy import median

from vseq.data import BaseDataset
from vseq.data.batchers import TextBatcher
from vseq.data.datapaths import PENN_TREEBANK_TRAIN
from vseq.data.token_map import TokenMap
from vseq.data.tokenizers import word_tokenizer
from vseq.data.transforms import EncodeInteger
from vseq.data.vocabulary import load_vocabulary


REPEATS = 5


class ResettingDataset():
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        """Reset the cached memory on every new for loop"""
        self.dataset.getitem.memory.clear()
        return iter(self.dataset)


vocab = load_vocabulary(PENN_TREEBANK_TRAIN)
token_map = TokenMap(tokens=vocab, add_start=False, add_end=False, add_delimit=True)
penn_treebank_transform = EncodeInteger(
    token_map=token_map,
    tokenizer=word_tokenizer,
)
batcher = TextBatcher()

modalities = [("txt", penn_treebank_transform, batcher)]


dataset_not_cached = BaseDataset(
    source=PENN_TREEBANK_TRAIN,
    modalities=modalities,
    cache=False,
)
dataset_cached = BaseDataset(
    source=PENN_TREEBANK_TRAIN,
    modalities=modalities,
    cache=True,
)
dataset_cached_reset = ResettingDataset(
    BaseDataset(
        source=PENN_TREEBANK_TRAIN,
        modalities=modalities,
        cache=True,
    )
)


def loop_dataset(dataset):
    for _ in dataset:
        pass


def report_timings(timings):
    min_t = min(timings)
    max_t = max(timings)
    median_t = sorted(timings)[len(timings) // 2 - 1]
    print(f"{min_t=:.3f}, {max_t=:.3f}, {median_t=:.3f}")
    return min_t, max_t, median_t


def time_dataset(dataset):
    timer = timeit.Timer("loop_dataset(dataset)", globals={"loop_dataset": loop_dataset, "dataset": dataset})
    timings = timer.repeat(repeat=5, number=1)
    min_t, max_t, median_t = report_timings(timings)
    return min_t, max_t, median_t


for i in range(REPEATS):
    print(f"        Non-cached {i}: ", end="")
    time_dataset(dataset_not_cached)

for i in range(REPEATS):
    print(f"Cached with resets {i}: ", end="")
    time_dataset(dataset_cached_reset)

for i in range(REPEATS):
    print(f"            Cached {i}: ", end="")
    time_dataset(dataset_cached)
