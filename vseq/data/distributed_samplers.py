from typing import Optional, Iterator, List

from operator import itemgetter

import torch

from torch.utils.data import Dataset, Sampler, DistributedSampler


"""
https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py

https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/26
"""


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)  # Calls self.sampler.__iter__
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """Wrapper over `Sampler` for distributed training. Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with `torch.nn.parallel.DistributedDataParallel`.
    In such case, each process can pass a `DistributedSamplerWrapper` instance as a `DataLoader` sampler, and load a
    subset of subsampled data of the original dataset that is exclusive to it.

    .. note::
        `Sampler` is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler: Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in distributed training
            rank (int, optional): Rank of the current process within ``num_replicas``
            shuffle (bool, optional): If true (default), sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        # Deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        self.epoch += 1

        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset

        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(
        self,
        sampler: Sampler,
        num_replicas: Optional[int],
        rank: Optional[int],
        shuffle: bool,
    ):
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)


class DistributedBatchSamplerWrapper(DistributedSampler):
    def __init__(
        self,
        sampler: Sampler,
        num_replicas: Optional[int],
        rank: Optional[int],
        shuffle: bool,
        seed: int,
        drop_last: bool,
    ) -> None:
        super().__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        raise NotImplementedError()

    def __iter__(self) -> Iterator[List[int]]:
        raise NotImplementedError()
