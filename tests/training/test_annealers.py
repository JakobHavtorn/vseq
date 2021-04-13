import pytest

from vseq.training.annealers import CosineAnnealer


@pytest.mark.parametrize(
    "n_steps,start_value,end_value",
    [
        (100, 0, 1),
        (100, 1, 0),
        (1, 0, 1),
        (100, 0.5, 0.5),
    ],
)
def test_cosine_annealer(n_steps, start_value, end_value):
    annealer = CosineAnnealer(n_steps=n_steps, start_value=start_value, end_value=end_value)

    assert annealer.value == start_value, "Must start at start value"

    for _ in range(n_steps - 1):

        value = annealer.value

        annealer.step()

        if start_value < end_value:
            assert annealer.value > value
        elif start_value > end_value:
            assert annealer.value < value
        else:
            assert annealer.value == value

    annealer.step()

    assert annealer.value == end_value, "Must end at end value"
