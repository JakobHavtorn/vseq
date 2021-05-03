import math

import pytest

from torch.utils.data import SequentialSampler, BatchSampler
from torch.utils.data.dataset import Dataset

from vseq.data.samplers import DistributedSamplerWrapper


class DummyDataset(Dataset):
    def __init__(self, n_examples: int = 20):
        self.examples = list(range(n_examples))

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)


def round_up_to_even(f):
    return math.ceil(f / 2.) * 2


def round_down_to_even(f):
    return math.floor(f / 2.) * 2


@pytest.mark.parametrize('n_examples, shuffle, drop_last',
    [
        (40, False, False),
        (20, False, False),
        (15, False, False),
        (40, True, False),
        (20, True, False),
        (15, True, False),
        (40, True, True),
        (20, True, True),
        (15, True, True),
        (40, False, True),
        (20, False, True),
        (15, False, True),
    ]
)
def test_distributed_sampler_wrapper_sequential(n_examples, shuffle, drop_last):
    num_replicas = 2
    sampler = SequentialSampler(DummyDataset(n_examples=n_examples))
    distributed_sampler = DistributedSamplerWrapper(
        sampler,
        rank=0,
        num_replicas=num_replicas,
        shuffle=shuffle,
        drop_last=drop_last
    )
    distributed_sampler2 = DistributedSamplerWrapper(
        sampler,
        rank=1,
        num_replicas=num_replicas,
        shuffle=shuffle,
        drop_last=drop_last
    )

    examples1 = [i for i in distributed_sampler]
    examples2 = [i for i in distributed_sampler2]

    assert len(examples1) == len(examples2)

    extra_samples_added = n_examples % num_replicas

    if drop_last:
        # if drop_last, examples beyond equal amounts in each sampler are dropped and there must be no overlap
        assert not any([i in examples2 for i in examples1])
    else:
        # if not drop_last, examples missing to get equal amonunts in each sampler are added from other samplers
        assert sum([i in examples2 for i in examples1]) == extra_samples_added

    if shuffle:
        # if shuffle, the examples should (on expectation) be differently ordered if sorted
        assert sorted(examples1) != examples1
        assert sorted(examples2) != examples2
    else:
        # if not shuffle, the examples should be sorted already (SequentialSampler)
        # (ignoring the potentially extra added samples)
        assert sorted(examples1[:-extra_samples_added]) == examples1[:-extra_samples_added]
        assert sorted(examples2[:-extra_samples_added]) == examples2[:-extra_samples_added]


@pytest.mark.parametrize('n_examples, batch_size, batch_drop_last, distributed_drop_last',
    [

        (40, 10, True,  False),  # 2 full batches for each worker
        (43, 10, True,  False),  # 2 full batches for each worker, one odd-sized batch left over
        (50, 10, True,  False),  # 2 full batches for each worker, one full-sized batch left over
        (53, 10, True,  False),  # 2 full batches for each worker, one full-sized and one odd-sized batch left over

        (40, 10, True,  True),
        (43, 10, True,  True),
        (50, 10, True,  True),
        (53, 10, True,  True),

        (40, 10, False, False),
        (43, 10, False, False),
        (50, 10, False, False),
        (53, 10, False, False),

        (40, 10, False, True),
        (43, 10, False, True),
        (50, 10, False, True),
        (53, 10, False, True),

    ]
)
def test_distributed_sampler_wrapper_sequential_batch(n_examples, batch_size, batch_drop_last, distributed_drop_last):
    num_replicas = 2
    sampler = BatchSampler(
        sampler=SequentialSampler(DummyDataset(n_examples=n_examples)),
        batch_size=batch_size,
        drop_last=batch_drop_last
    )
    distributed_sampler = DistributedSamplerWrapper(
        sampler,
        rank=0,
        num_replicas=num_replicas,
        drop_last=distributed_drop_last
    )
    distributed_sampler2 = DistributedSamplerWrapper(
        sampler,
        rank=1,
        num_replicas=num_replicas,
        drop_last=distributed_drop_last
    )

    max_batches = math.ceil(n_examples / batch_size)
    # extra_examples = n_examples % batch_size

    total_examples_to_drop = (n_examples % (batch_size * num_replicas))
    full_batches_to_drop = total_examples_to_drop // batch_size
    total_batches_to_drop = math.ceil(total_examples_to_drop / batch_size)
    size_odd_batch_to_drop = total_examples_to_drop - full_batches_to_drop * batch_size
    print(f"\n{max_batches=}\n{total_examples_to_drop=}\n{full_batches_to_drop=}\n{total_batches_to_drop=}\n{size_odd_batch_to_drop=}")

    batches1 = [i for i in distributed_sampler]
    batches2 = [i for i in distributed_sampler2]

    assert len(batches1) == len(batches2), "Must always be same number of batches1"

    print(batches1, batches2)

    if batch_drop_last and not distributed_drop_last:
        # batch sampler drops last batch but distributed sampler creates a new batch with copies from other subset
        assert len(batches1[-1]) == len(batches2[-1])
        # assert batches1[0] == batches2[-1], "the new batch in `batches2` is the first batch in `batches1`"
    elif batch_drop_last and distributed_drop_last:
        # batch sampler drops last batch and distributed sampler removes superfluous batch of the remaining process.
        # result is between 0 and and `num_replicas` batches1 dropped (corresponds to rounding to nearest even below)
        assert 2 * len(batches1) == max_batches - total_batches_to_drop
    elif not batch_drop_last and not distributed_drop_last:
        # batch sampler keeps the odd sized batch and the distributed sampler removes superfluous batch of the remaining process
        # result is between 0 and `num_replicas - 1` batches1 added (corresponds to rounding to nearest even above).
        assert len(batches1) == round_up_to_even(max_batches) // num_replicas
        final_odd_batch_size = n_examples % batch_size
        if final_odd_batch_size:
            assert (len(batches1[-1]) == final_odd_batch_size) or (len(batches2[-1]) == final_odd_batch_size)
    elif not batch_drop_last and distributed_drop_last:
        # batch sampler keeps the odd sized batch and distributed sampler creates a new batch with copies from other subset
        if not total_batches_to_drop and size_odd_batch_to_drop:
            assert (len(batches1[-1]) == size_odd_batch_to_drop) or (len(batches2[-1]) == size_odd_batch_to_drop)
