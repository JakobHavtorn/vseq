import pytest

import torch

from vseq.modules import WordDropout


@pytest.mark.parametrize("dropout_rate, x_shape", [
    (0.25, (4, 10000)),
    (0.75, (4, 10000, 8)),
    (0.50, (4, 10000, 2, 4)),
    (0.50, (4, 10000, 2, 4, 5)),
])
def test_word_dropout_rate(dropout_rate, x_shape):
    dropout = WordDropout(dropout_rate=dropout_rate, mask_value=0)
    x = torch.ones(x_shape)
    y = dropout(x)
    
    assert dropout_rate - 0.01 < 1 - y.mean() < dropout_rate + 0.01
    

def test_word_dropout_all_features_per_dropped_timestep():
    torch.manual_seed(0)
    dropout = WordDropout(dropout_rate=0.5, mask_value=0)
    x = torch.ones((1, 10, 5))
    y = dropout(x)

    assert sorted(torch.unique((y == 0).sum(2)).tolist()) == [0, 5], "Zero or all 5 dimensions must have been masked"
