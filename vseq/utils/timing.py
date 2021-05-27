import warnings
import timeit as timeit_module

from types import SimpleNamespace

from typing import Callable, Optional, Union

import numpy as np


UNITS = {"nsec": 1e-9, "usec": 1e-6, "msec": 1e-3, "sec": 1.0}


def format_time(dt, unit="sec", precision=3):
    if unit is not None:
        scale = UNITS[unit]
    else:
        scales = [(scale, unit) for unit, scale in UNITS.items()]
        scales.sort(reverse=True)
        for scale, unit in scales:
            if dt >= scale:
                break

    return "%.*g %s" % (precision, dt / scale, unit)


def timeit(
    statement: Union["str", Callable],
    setup: str = "pass",
    timer=timeit_module.default_timer,
    globals: dict = None,
    inner_duration: float = 0.2,
    repeats: int = 10
):
    r"""Time the execution of `statement` using the `timeit` package similar to the IPython magic `%timeit`.

    Example::

        import random

        def a():
            return random.random() + random.random()

        timeit(a)

        >>> namespace(min=1.844983547925949e-07,
                      max=2.3509945720434188e-07,
                      mean=1.979972431436181e-07,
                      median=1.9188937079161407e-07,
                      std=1.4883481745944718e-08,
                      number=1000000,
                      repeats=10)

    Args:
        statement (Union[str, Callable]): Statement to `exec` or a callable.
        setup (str, optional): Any setup steps to perform before repeats. Defaults to "pass".
        timer (optional): Timer to use. Defaults to timeit_module.default_timer.
        globals (dict, optional): Namespace for timing. Defaults to None.
        inner_duration (float, optional): Minimum number of seconds to use iterating over `statement`. Defaults to 0.2.
        repeats (int, optional): Number of times to repeat the inner loop.

    Returns:
        SimpleNamespace: Namespace with min, max, mean, median, std, number and repeats attributes
    """
    timer = timeit_module.Timer(statement, setup=setup, timer=timer, globals=globals)

    # Autorange twice to overcome overhead
    number, time_taken = timer.autorange()
    number, time_taken = timer.autorange()

    multiplier = inner_duration // time_taken + 1
    number = int(multiplier * number)

    # Time
    timings = timer.repeat(repeat=repeats, number=number)
    timings = np.array(timings) / number

    # Collect results
    min = np.min(timings)
    max = np.max(timings)
    mean = np.mean(timings)
    median = np.median(timings)
    std = np.std(timings)

    if max >= min * 4:
        warnings.warn_explicit(
            "The test results are likely unreliable. "
            "The worst time (%s) was more than four times "
            "slower than the best time (%s)." % (format_time(max), format_time(min)),
            UserWarning,
            "",
            0,
        )

    return SimpleNamespace(min=min, max=max, mean=mean, median=median, std=std, number=number, repeats=repeats)


def report_timings(timings, prefix: Optional[str] = None):
    if prefix:
        s = f"{prefix:15s} | "
    else:
        s = ""
        
    s += f"number={timings.number:5d} | [{timings.min:.3e}, {timings.max:.3e}] | {timings.median:.3e} | {timings.mean:.3e} +- {timings.std:.3e} s"
    print(s)
    return s
