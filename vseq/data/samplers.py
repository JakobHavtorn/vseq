import random

from typing import Iterator, Union, List
from torch.utils.data.sampler import Sampler

import numpy as np

from .datapaths import DATAPATHS_MAPPING


class FrameSampler(Sampler):
    def __init__(
        self,
        source: str,
        sample_rate: int,
        max_seconds: float = 320.0,
        max_pool_difference: float = 0.3,
        min_pool_size: int = 512,
        num_batches: Union[int, None] = None,
    ):
        """
        The sampler groups the source into sample pools of examples with similar length meeting criterias defined by
        'max_pool_difference' and 'min_pool_size'. Batches of close to, but never more than, 'max_seconds', are
        constructed by first sampling a pool and then sampling each batch from from within that pool.

        Args:
            source (object): Dataset for which the sampler will be used.
            sample_rate (int): Used for converting the length of the PCM file to seconds.
            max_seconds (float): The maximum size of the batch in seconds.
            max_pool_difference (float): The maximum length difference between shortest and longest sample a pool.
            min_pool_size (float): The minimum number of examples in a pool. Overwrites max_pool_difference.
            num_batches (int or None): Samples num_batches (with replacement if necessary) instead of running a standard epoch.
        """

        self.source = source
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.max_pool_difference = max_pool_difference
        self.min_pool_size = min_pool_size
        self.num_batches = num_batches
        self.buffer = [] # only used when num_batches is not None

        self.source_filepath = DATAPATHS_MAPPING[source] if source in DATAPATHS_MAPPING else source
        self.lengths = self.load_lengths(self.source_filepath)
        self.pools = self.create_sample_pools(max_pool_difference, min_pool_size)
        self.batches = self.sample_batches()

        assert self.lengths.max() < self.max_seconds, "One or more examples are longer than the maximum number of seconds per batch."

    def load_lengths(self, source_filepath):
        """
        Loads the example lengths into an array with same order as the examples of the source dataset.
        """
        with open(source_filepath, "r") as source_file_buffer:
            lines = source_file_buffer.read().splitlines()

        lengths = [int(l.split(",")[1]) / self.sample_rate for l in lines]
        return np.array(lengths)

    def create_sample_pools(self, max_diff, min_size):
        """
        Creates the sample pools. Can be used to change to the sampling criteria without creating a new sampler.
        """
        start, end = 0, 0
        sorted_idxs = np.argsort(self.lengths)
        sorted_lens = self.lengths[sorted_idxs]

        pools = []
        while end != len(self.lengths):
            base_len = sorted_lens[start]
            deltas = sorted_lens - base_len
            pool_size = np.logical_and(0 <= deltas, deltas < max_diff).sum()
            end = min(max(start + min_size, start + pool_size), len(self.lengths))
            if (len(self.lengths) - end) < min_size:
                end = len(self.lengths)

            pools.append(sorted_idxs[start:end].tolist())
            start = end

        return pools

    def sample_batches(self):
        """
        Sample batches from the pools.
        """

        if self.num_batches is not None:
            if len(self.buffer) >= self.num_batches:
                batches = self.buffer[:self.num_batches]
                self.buffer = self.buffer[self.num_batches:]
                return batches

        ordered_idxs = np.concatenate([random.sample(p, k=len(p)) for p in self.pools])  # shuffle each pool internally
        batch_idxs = (self.lengths[ordered_idxs].cumsum() // self.max_seconds).astype(int)
        split_points = np.bincount(batch_idxs).cumsum()[:-1] # the last split is implicit
        batches = np.array_split(ordered_idxs, split_points)
        batches = list(map(lambda x: x.tolist(), batches))
        random.shuffle(batches)  # shuffle the order of batches

        if self.num_batches is not None:
            self.buffer += batches
            return self.sample_batches()

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        try:
            for batch in self.batches:
                yield batch
        finally:
            self.batches = self.sample_batches()  # to ensure batches are resampled if interrupted

    def __len__(self):
        return len(self.batches)


class EvalSampler(Sampler):
    def __init__(
        self,
        source: str,
        sample_rate: int,
        max_seconds: float = 320.0
    ):
        """
        The sampler groups the source into sample pools of examples with similar length meeting criterias defined by
        'max_pool_difference' and 'min_pool_size'. Batches of up to 'seconds' are constructed by sampling from
        one pool at the time.

        Args:
            source (object): Dataset for which the sampler will be used.
            sample_rate (int): Used for converting the length of the PCM file to seconds.
            max_seconds (float): The maximum size of the batch in seconds.
        """

        self.source = source
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate

        self.source_filepath = DATAPATHS_MAPPING[source] if source in DATAPATHS_MAPPING else source
        self.lengths = self.load_lengths(self.source_filepath)
        self.batches = self.sample_batches()

    def load_lengths(self, source_filepath):
        """
        Loads the example lengths into an array with same order as the examples of the source dataset.
        """
        with open(source_filepath, "r") as source_file_buffer:
            lines = source_file_buffer.read().splitlines()

        lengths = [int(l.split(",")[1]) / self.sample_rate for l in lines]
        return np.array(lengths)

    def sample_batches(self):
        """
        Sample batches from the pools.
        """

        sorted_idxs = np.argsort(self.lengths)
        batch_idxs = (self.lengths[sorted_idxs].cumsum() // self.max_seconds).astype(int)
        split_points = np.bincount(batch_idxs).cumsum()[:-1] # the last split is implicit
        batches = np.array_split(sorted_idxs, split_points)
        batches = list(map(lambda x: x.tolist(), batches))
        return batches
    
    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
