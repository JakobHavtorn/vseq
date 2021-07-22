import pytest

import uniplot

from vseq.training.annealers import CosineAnnealer


INF = float("inf")


@pytest.mark.parametrize(
    "anneal_steps, constant_steps, start_value, end_value",
    [
        (100,  0,    0,    1),
        (100,  0,    1,    0),
        (100,  0,    1,    1),
        (  1,  0,    0,    1),
        (  0,  0,    0,    1),
        (  0,  0,    1,    1),
        (100,  0,   -1,    2),
        (100,  0,    1,   -2),

        (10,    5,    0,    1),
        (100,  50,    1,    0),
        (100,  50,    1,    1),
        (  1,   1,    0,    1),
        (  0,   0,    0,    1),
        (  0,   0,    1,    1),
        (100,  50,   -1,    2),
        (100,  50,    1,   -2),
    ],
)
def test_cosine_annealer_function(anneal_steps, constant_steps, start_value, end_value):
    annealer = CosineAnnealer(
        anneal_steps=anneal_steps,
        constant_steps=constant_steps,
        start_value=start_value,
        end_value=end_value,
    )

    assert annealer.value == None, "Must start at None when not yet `step()`ed"

    constant_values = [annealer.step() for _ in range(constant_steps)]
    assert all(v == annealer.start_value for v in constant_values), "During constant_steps the value must remain constant"

    anneal_values = [annealer.step() for _ in range(anneal_steps)]
    anneal_diffs = [anneal_values[i + 1] - anneal_values[i] for i in range(anneal_steps - 1)]
    if start_value < end_value:
        assert all(v > 0 for v in anneal_diffs)
    elif start_value > end_value:
        assert all(v < 0 for v in anneal_diffs)
    else:
        assert all(v == 0 for v in anneal_diffs)
    
    if anneal_steps > 0 or constant_steps > 0:
        assert annealer.value == end_value, "Must end at end value"
    else:
        assert annealer.value == None, "Must end at None when not yet `step()`ed"

    end_values = [annealer.step() for _ in range(10)]
    assert all(v == end_value for v in end_values), "steps after reaching anneal_steps all return end_value"


@pytest.mark.parametrize(
    "anneal_steps, constant_steps, start_value, end_value",
    [
        (100,  0,   0,    INF),
        (100,  0,   INF,  0),
        (100,  0,   INF,  INF),
        (-1,   0,   0,    1),
        (1,   -1,   0,    1),
    ],
)
def test_cosine_annealer_failure(anneal_steps, constant_steps, start_value, end_value):
    with pytest.raises(Exception):
        CosineAnnealer(
            anneal_steps=anneal_steps,
            constant_steps=constant_steps,
            start_value=start_value,
            end_value=end_value,
        )


def test_cosine_annealer_plot():
    annealer1 = CosineAnnealer(anneal_steps=200, constant_steps=0, start_value=0, end_value=1)
    values1 = [annealer1.step() for _ in range(200)]

    annealer2 = CosineAnnealer(anneal_steps=100, constant_steps=100, start_value=0, end_value=2)
    values2 = [annealer2.step() for _ in range(200)]

    annealer3 = CosineAnnealer(anneal_steps=0, constant_steps=0, start_value=0, end_value=1.5)
    values3 = [annealer3.step() for _ in range(200)]

    print()
    xs = list(range(200))
    uniplot.plot([values1, values2, values3], [xs, xs, xs], legend_labels=[str(annealer1), str(annealer2), str(annealer3)])
