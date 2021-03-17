import os
import itertools
import random


import numpy as np
from typing import Union
from torch.utils.data.sampler import Sampler

from .base_dataset import BaseDataset
from .datapaths import DATAPATHS_MAPPING

class FrameSampler(Sampler):

    def __init__(
        self,
        source: str,
        sample_rate: int,
        max_seconds: float = 320.0,
        max_pool_difference: float = 0.3,
        min_pool_size: int = 512,
        num_batches: Union[int, None] = None):
        """
        The sampler groups the dataset into sample pools of examples with similar length meeting criterias defined by
        'max_sample_pool_diff' and 'min_sample_pool_size'. Batches of up to 'seconds' are constructed by sampling from
        one pool at the time.

        Args:
            dataset (object): Dataset for which the sampler will be used.
            sample_rate (int): Used for converting the length of the PCM file to seconds.
            max_seconds (float): The maximum size of the batch in seconds.
            max_sample_pool_diff (float): The maximum length difference between shortest and longest sample a pool.
            min_sample_pool_size (float): The minimum number of examples in a pool. Overwrites max_sample_pool_diff.
            num_batches (int or None): Samples num_batches with replacement instead of running a standard epoch.
        """

        self.source = source
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.max_pool_difference = max_pool_difference
        self.min_pool_size = min_pool_size
        self.num_batches = num_batches
        
        self.source_filepath = DATAPATHS_MAPPING[source] if source in DATAPATHS_MAPPING else source
        self.lengths = self.load_lengths(self.source_filepath)
        self.pools = self.create_sample_pools(max_pool_difference, min_pool_size)
        self.batches = self.sample_batches()
        
    def load_lengths(self, source_filepath):
        """
        Loads the example lengths into an array with same order as the examples of the source dataset.
        """
        with open(source_filepath, "r") as source_file_buffer:
            lines = source_file_buffer.read().splitlines()
        return np.array([int(l.split(",")[1]) / self.sample_rate for l in lines])

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
            deltas = (sorted_lens - base_len)
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

        ordered_example_idxs = list(itertools.chain(*[random.sample(p, k=len(p)) for p in self.pools]))

        batches = []
        first_idx = ordered_example_idxs[0]
        batch = [first_idx]
        batch_sequence_length = self.lengths[first_idx]

        for idx in ordered_example_idxs[1:]:
            batch_sequence_length = max(batch_sequence_length, self.lengths[idx])
            if (len(batch) + 1) * batch_sequence_length < self.max_seconds:
                batch.append(idx)
            else:
                batches.append(batch)
                batch = [idx]
                batch_sequence_length = self.lengths[idx]
        batches.append(batch)
        random.shuffle(batches)

        if self.num_batches is not None:
            if len(batches) > self.num_batches:
                batches = random.sample(batches, k=self.num_batches)
            else:
                batches = random.choices(batches, k=self.num_batches)

        return batches

    def __iter__(self):
        try:
            for batch in self.batches:
                yield batch
            self.batches = self.sample_batches()
        except:
            self.batches = self.sample_batches() # to ensure batches are resampled if interrupted
        
    def __len__(self):
        return len(self.batches)

# class EvalSampler(Sampler):

#     def __init__(self, dataset, max_seconds=320.0, sample_rate=16000):
#         """
#         Examples are sorted by sequence length and batches constructed by adding up to max_seconds in each batch
#         starting from shortest to longest example.

#         Args:
#             dataset (object): Dataset for which the sampler will be used.
#             sample_rate (int): Used for converting the length of the PCM file to seconds.
#             max_seconds (float): The maximum size of the batch in seconds.
#         """
#         self.dataset = dataset
#         self.max_seconds = max_seconds
#         self.sample_rate = sample_rate
#         self.lens = None # set by load_lengths
#         self.batches = None # set by sample_batches
        
#         self.load_lengths()
#         self.sample_batches()

#     def load_lengths(self):
#         """
#         Loads the example lengths into an array with same order as the examples of the source dataset.
#         """
#         with open(self.dataset.source, 'r') as f:
#             lines = f.read().splitlines()
#         self.lens = np.array([int(l.split(',')[1]) / self.sample_rate for l in lines])

#     def sample_batches(self):
#         """
#         Sample batches in ascending order.
#         """
#         batches, batch, batch_len = [], [], 0
#         sorted_idxs = np.argsort(self.lens)
#         for i in sorted_idxs:
#             batch_len = max(batch_len, self.lens[i])
#             if batch_len * (len(batch) + 1) < self.max_seconds:
#                 batch.append(i)
#             else:
#                 batches.append(batch)
#                 batch = [i]
#         batches.append(batch)
#         self.batches = batches

#     def __iter__(self):
#         for batch in self.batches:
#             yield batch
        
#     def __len__(self):
#         return len(self.batches)