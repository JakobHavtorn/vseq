import numpy as np
import torch

from vseq.utils.operations import reverse_sequences


def get_x(batch_first: bool = False):
    x = torch.stack(
        [
            torch.cat([1 + torch.arange(10), torch.zeros(0)]),
            torch.cat([1 + torch.arange(7), torch.zeros(3)]),
            torch.cat([1 + torch.arange(5), torch.zeros(5)]),
            torch.cat([1 + torch.arange(2), torch.zeros(8)]),
        ],
        dim=1,
    )
    x_rev = torch.stack(
        [
            torch.cat([1 + torch.arange(10).flip(0), torch.zeros(0)]),
            torch.cat([1 + torch.arange(7).flip(0), torch.zeros(3)]),
            torch.cat([1 + torch.arange(5).flip(0), torch.zeros(5)]),
            torch.cat([1 + torch.arange(2).flip(0), torch.zeros(8)]),
        ],
        dim=1,
    )
    x_sl = torch.tensor([10, 7, 5, 2], dtype=int)

    if batch_first:
        x = x.permute(1, 0)
        x_rev = x_rev.permute(1, 0)

    return x, x_sl, x_rev


def test_reverse_sequences():
    x, x_sl, x_rev = get_x()

    x_rev_computed = reverse_sequences(x, x_sl, batch_first=False)

    np.testing.assert_allclose(x_rev, x_rev_computed)


def test_reverse_sequences_batch_first():
    x, x_sl, x_rev = get_x(batch_first=True)

    x_rev_computed = reverse_sequences(x, x_sl, batch_first=True)

    np.testing.assert_allclose(x_rev, x_rev_computed)
