import warnings
import timeit as timeit_module

from types import SimpleNamespace

from typing import Callable, Optional, Union

import numpy as np


UNITS = {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0}


def format_time(dt, unit=None, precision=4):
    """Format a time in seconds by rescaling and appending appropriate unit
    
    The returned string will always have length at most 5 (precision) + 2 (unit) + 1 (space) = 7
    """
    if unit is not None:
        scale = UNITS[unit]
    else:
        scales = [(scale, unit) for unit, scale in UNITS.items()]
        scales.sort(reverse=True)
        for scale, unit in scales:
            if dt >= scale:
                break

    return "%.*g%s" % (precision, dt / scale, unit)


def timeit(
    statement: Union["str", Callable],
    setup: str = "pass",
    timer=timeit_module.default_timer,
    globals: dict = None,
    inner_duration: float = 0.2,
    repeats: int = 10,
    number: Optional[int] = None,
    print_results: bool = False
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
        number (int, optional): Overrules the `inner_duration` argument and directly sets the `number` of inner iters.

    Returns:
        SimpleNamespace: Namespace with min, max, mean, median, std, number and repeats attributes
    """
    timer = timeit_module.Timer(statement, setup=setup, timer=timer, globals=globals)

    if number is None:
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
    
    results = SimpleNamespace(min=min, max=max, mean=mean, median=median, std=std, number=number, repeats=repeats)

    if max >= min * 4:
        warnings.warn_explicit(
            "The test results are likely unreliable. "
            "The worst time (%s) was more than four times "
            "slower than the best time (%s)." % (format_time(max), format_time(min)),
            UserWarning,
            "",
            0,
        )

    if print_results:
        report_timings(results, suffix=str(statement))

    return results


def report_timings(timings, prefix: Optional[str] = None, suffix: Optional[str] = None):
    if prefix:
        s = f"{prefix:15s} | "
    else:
        s = ""
        
    s += f"number={timings.number:>5d} | [{format_time(timings.min):>8s}, {format_time(timings.max):>8s}] | {format_time(timings.median):>8s} | {format_time(timings.mean):>8s} +- {format_time(timings.std):>8s}"

    if suffix:
        s += f" || {suffix}"

    print(s)
    return s
