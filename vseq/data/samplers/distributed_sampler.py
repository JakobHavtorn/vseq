import math
import random

from typing import Optional, Iterator, List

from operator import itemgetter

import torch
import torch.distributed as distributed
import numpy as np

from torch.utils.data import Dataset, Sampler, DistributedSampler


# class DistributedSampler(Sampler):
#     r"""Sampler that restricts data loading to a subset of the dataset.

#     It is especially useful in conjunction with
#     :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
#     process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
#     :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
#     original dataset that is exclusive to it.

#     # .. note::
#     #     Dataset is assumed to be of constant size.

#     Args:
#         dataset: Dataset used for sampling.
#         num_replicas (int, optional): Number of processes participating in
#             distributed training. By default, :attr:`world_size` is retrieved from the
#             current distributed group.
#         rank (int, optional): Rank of the current process within :attr:`num_replicas`.
#             By default, :attr:`rank` is retrieved from the current distributed
#             group.
#         shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
#             indices.
#         seed (int, optional): random seed used to shuffle the sampler if
#             :attr:`shuffle=True`. This number should be identical across all
#             processes in the distributed group. Default: ``0``.
#         drop_last (bool, optional): if ``True``, then the sampler will drop the
#             tail of the data to make it evenly divisible across the number of
#             replicas. If ``False``, the sampler will add extra indices to make
#             the data evenly divisible across the replicas. Default: ``False``.

#     .. warning::
#         In distributed mode, calling the :meth:`set_epoch` method at
#         the beginning of each epoch **before** creating the :class:`DataLoader` iterator
#         is necessary to make shuffling work properly across multiple epochs. Otherwise,
#         the same ordering will be always used.

#     Example::

#         >>> sampler = DistributedSampler(dataset) if is_distributed else None
#         >>> loader = DataLoader(dataset, shuffle=(sampler is None),
#         ...                     sampler=sampler)
#         >>> for epoch in range(start_epoch, n_epochs):
#         ...     if is_distributed:
#         ...         sampler.set_epoch(epoch)
#         ...     train(loader)
#     """

#     def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
#                  rank: Optional[int] = None, shuffle: bool = True,
#                  seed: int = 0, drop_last: bool = False) -> None:
#         if num_replicas is None:
#             if not distributed.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             num_replicas = distributed.get_world_size()
#         if rank is None:
#             if not distributed.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             rank = distributed.get_rank()
#         if rank >= num_replicas or rank < 0:
#             raise ValueError(
#                 "Invalid rank {}, rank should be in the interval"
#                 " [0, {}]".format(rank, num_replicas - 1))
#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.drop_last = drop_last
#         # If the dataset length is evenly divisible by # of replicas, then there
#         # is no need to drop any data, since the dataset will be split equally.
#         if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
#             # Split to nearest available length that is evenly divisible.
#             # This is to ensure each rank receives the same amount of data when
#             # using this Sampler.
#             self.num_samples = math.ceil(
#                 # `type:ignore` is required because Dataset cannot provide a default __len__
#                 # see NOTE in pytorch/torch/utils/data/sampler.py
#                 (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
#             )
#         else:
#             self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
#         self.total_size = self.num_samples * self.num_replicas
#         self.shuffle = shuffle
#         self.seed = seed

#     def __iter__(self) -> Iterator:
#         # # If the dataset length is evenly divisible by # of replicas, then there
#         # # is no need to drop any data, since the dataset will be split equally.
#         # if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
#         #     # Split to nearest available length that is evenly divisible.
#         #     # This is to ensure each rank receives the same amount of data when
#         #     # using this Sampler.
#         #     self.num_samples = math.ceil(
#         #         # `type:ignore` is required because Dataset cannot provide a default __len__
#         #         # see NOTE in pytorch/torch/utils/data/sampler.py
#         #         (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
#         #     )
#         # else:
#         #     self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
#         # total_size = self.num_samples * self.num_replicas

#         if self.shuffle:
#             # deterministically shuffle based on epoch and seed
#             g = torch.Generator()
#             g.manual_seed(self.seed + self.epoch)
#             indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
#         else:
#             indices = list(range(len(self.dataset)))  # type: ignore

#         if not self.drop_last:
#             # add extra samples to make it evenly divisible
#             padding_size = self.total_size - len(indices)
#             if padding_size <= len(indices):
#                 indices += indices[:padding_size]
#             else:
#                 indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
#         else:
#             # remove tail of data to make it evenly divisible.
#             indices = indices[:self.total_size]
#         assert len(indices) == self.total_size

#         # subsample
#         indices = indices[self.rank:self.total_size:self.num_replicas]
#         assert len(indices) == self.num_samples

#         return iter(indices)

#     def __len__(self) -> int:
#         return self.num_samples

#     def set_epoch(self, epoch: int) -> None:
#         r"""
#         Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
#         use a different random ordering for each epoch. Otherwise, the next iteration of this
#         sampler will yield the same ordering.

#         Args:
#             epoch (int): Epoch number.
#         """
#         self.epoch = epoch


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
            # Call self.sampler.__iter__ and iterate through it
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """Wrapper of `Sampler` for distributed training. Allows use of any (constant size) sampler in distributed mode.

    It is especially useful in conjunction with `torch.nn.parallel.DistributedDataParallel`. Each process can then
    pass a `DistributedSamplerWrapper` instance to a `DataLoader` as `sampler` or `batch_sampler`.

    On every epoch, each process seeds torch, numpy and random according to the epoch and `seed` and resamples the 
    entire dataset using the `sampler`. This results in `num_replicas` identical samplings of the dataset.

    This sampling of the dataset can additionally be randomly shuffled (`shuffle==True`) before being divided into
    subsets. This is useful if the `sampler` is deterministic and we want the unique subset passed to each process to
    be different for every epoch. Otherwise, it rarely harms.

    If the dataset is not evenly divisible among `num_replicas` processes, the distributed sampler either adds
    additional copies (if `drop_last == False`) or removes the extra number of examples (if `drop_last == True`).

    The DistributedSamplerWrapper then instantiates a DistributedSampler to deterministically define `num_replicas`
    different subsets of this sampling of the dataset.

    The end-result is `num_replicas` equally sized and (almost) unique subsets of the dataset as sampled by `sampler`.

    This works both when `sampler` is a `Sampler` and a `BatchSampler`.

    If batch sampling, the `drop_last` argument ...

    Sources:
        https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
        https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/26
    """

    def __init__(
        self,
        sampler: Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: Optional[bool] = False,
        seed: Optional[int] = 0,
        drop_last: Optional[bool] = False
    ):
        """
        Args:
            sampler: Sampler used to sample some dataset (`Sampler` or `BatchSampler`)
            num_replicas (int, optional): Number of processes participating in
                distributed training. By default, :attr:`world_size` is retrieved from the
                current distributed group.
            rank (int, optional): Rank of the current process within :attr:`num_replicas`.
                By default, :attr:`rank` is retrieved from the current distributed
                group.
            shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
                dataset indices such that the unique subsets assigned to the processes
                are not the same for every epoch.
            seed (int, optional): random seed used to shuffle the sampler if
                :attr:`shuffle=True`. This number should be identical across all
                processes in the distributed group. Default: ``0``.
            drop_last (bool, optional): if ``True``, then the sampler will drop the
                tail of the data to make it evenly divisible across the number of
                replicas. If ``False``, the sampler will add extra indices to make
                the data evenly divisible across the replicas. Default: ``False``.
        """
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.dataset = DatasetFromSampler(self.sampler)

        # super().__init__(
        distributed_sampler = DistributedSampler(
            dataset=self.dataset,
            num_replicas=self.num_replicas,
            rank=self.rank,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last
        )
        self.num_samples = distributed_sampler.num_samples
        self.epoch = 0

    # def __iter__(self) -> Iterator[int]:
    #     """Iterate over sampler.

    #     Returns:
    #         python iterator
    #     """
    #     # Deterministically shuffle based on epoch to get distinct splits for each worker
    #     torch.manual_seed(self.seed + self.epoch)
    #     random.seed(self.seed + self.epoch)
    #     np.random.seed(self.seed + self.epoch)
    #     self.epoch += 1

    #     self.dataset = DatasetFromSampler(self.sampler)

    #     self.ds = DistributedSampler(
    #         DatasetFromSampler(self.sampler),
    #         num_replicas=self.num_replicas,
    #         rank=self.rank,
    #         shuffle=self.shuffle,
    #         seed=self.seed,
    #         drop_last=self.drop_last
    #     )
    #     self.num_samples = self.ds.num_samples
    #     indices_of_indices = self.ds.__iter__()  # Get subsampled indices for this process
    #     # return the indices given by `indices_of_indices` from the dataset
    #     return iter(itemgetter(*indices_of_indices)(self.dataset))

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        # Deterministically shuffle based on epoch to get distinct splits for each worker
        torch.manual_seed(self.seed + self.epoch)
        random.seed(self.seed + self.epoch)
        np.random.seed(self.seed + self.epoch)
        self.epoch += 1

        self.dataset = DatasetFromSampler(self.sampler)

        distributed_sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.num_replicas,
            rank=self.rank,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last
        )
        self.num_samples = distributed_sampler.num_samples
        indices_of_indices = distributed_sampler.__iter__()  # Get subsampled indices for this process
        # return the indices given by `indices_of_indices` from the dataset
        return iter(itemgetter(*indices_of_indices)(self.dataset))

    # def __iter__(self) -> Iterator[int]:
    #     """Iterate over sampler.

    #     Returns:
    #         python iterator
    #     """
    #     # Deterministically shuffle based on epoch to get distinct splits for each worker
    #     torch.manual_seed(self.seed + self.epoch)
    #     random.seed(self.seed + self.epoch)
    #     np.random.seed(self.seed + self.epoch)

    #     self.epoch += 1

    #     self.dataset = DatasetFromSampler(self.sampler)  # Resample entire dataset identically across processes
    #     indices_of_indices = super().__iter__()  # Get subsampled indices for this process
    #     # return the indices given by `indices_of_indices` from the dataset
    #     return iter(itemgetter(*indices_of_indices)(self.dataset))

    def __repr__(self):
        sampler = self.sampler
        num_replicas = self.num_replicas
        rank = self.rank
        shuffle = self.shuffle
        seed = self.seed
        drop_last = self.drop_last
        s = f"DistributedSamplerWrapper({sampler=}, {num_replicas=}, {rank=}, {shuffle=}, {seed=}, {drop_last=})"
        return s
