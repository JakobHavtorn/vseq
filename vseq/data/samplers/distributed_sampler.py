from typing import Optional, Iterator, List

from operator import itemgetter

import torch

from torch.utils.data import Dataset, Sampler, DistributedSampler


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
    pass a `DistributedSamplerWrapper` instance as a `DataLoader` `sampler` or `batch_sampler`.

    On every epoch, each process seeds torch according to the epoch and `seed` and resamples the entire dataset
    using the `sampler`. This results in `num_replicas` identical samplings of the dataset.

    This sampling of the dataset can additionally be randomly shuffled (`shuffle==True`) before being divided into
    subsets. This is useful if the `sampler` is deterministic and we want the unique subset passed to each process to
    be different for every epoch. Otherwise, it doesn't harm.

    If the dataset is not evenly divisible among `num_replicas` processes, the distributed sampler either adds
    additional copies (`drop_last == False`) or removes the extra number of examples (`drop_last == True`).

    The DistributedSamplerWrapper then uses the DistributedSampler to deterministically define `num_replicas`
    different subsets of this sampling of the dataset.
    
    The end-result is `num_replicas` equally sized and (almost) unique subsets of the dataset as sampled by `sampler`.

    This works both when `sampler` is a `Sampler` and a `BatchSampler`.
    
    If batch sampling, the `drop_last` argument ...

    .. note::
        `sampler` is assumed to be of constant size.

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
        super().__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        # Deterministically shuffle based on epoch to get distinct splits for each worker
        torch.manual_seed(self.seed + self.epoch)
        self.epoch += 1

        self.dataset = DatasetFromSampler(self.sampler)  # Resample entire dataset identically across processes
        indices_of_indices = super().__iter__()  # Get subsampled indices for this process
        # return the indices given by `indices_of_indices` from the dataset
        return iter(itemgetter(*indices_of_indices)(self.dataset))
