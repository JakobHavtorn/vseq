import math


class Annealer:
    """Abstract base class for annealers

    All annealers should have their initial `value` set to `None` after `__init__()`.
    On the first call to `step()`, the `start_value` is returned.
    On every subsequent call to `step()`, the `value` is (potentially) incremented and the (new) `value` is returned.
    """

    value = None

    def step(self) -> float:
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class CosineAnnealer(Annealer):
    """Anneal a `value` using a cosine annealing schedule as in [1].

    Args:
        anneal_steps (int): Number of steps to go from `min_value` to `max_value`
        constant_steps (int): Number of steps to remain at `start_value` before annealing.
        start_value (float): Value to return before first `step()`. Default: 0.
        end_value (float): Value to return after the `anneal_steps`th `step()`. Default: 1.

    [1] SGDR: Stochastic Gradient Descent with Warm Restarts: https://arxiv.org/abs/1608.03983
    """

    def __init__(self, anneal_steps: int, constant_steps: int = 0, start_value: float = 0, end_value: float = 1):
        super().__init__()

        self.validate_inputs(anneal_steps, constant_steps, start_value, end_value)

        self.anneal_steps = anneal_steps
        self.constant_steps = constant_steps
        self.start_value = start_value
        self.end_value = end_value
        self.steps = 0
        self.value = None

    @staticmethod
    def validate_inputs(anneal_steps, constant_steps, start_value, end_value):
        if anneal_steps < 0 or constant_steps < 0:
            raise ValueError(f"steps must be positive but got {anneal_steps=}, {constant_steps=}")
        if not math.isfinite(start_value) or not math.isfinite(end_value):
            raise ValueError(f"start_value and end_value must be finite but got {start_value=}, {end_value=}")

    def step(self):
        self.steps += 1

        if self.steps >= self.anneal_steps + self.constant_steps:
            self.value = self.end_value
        elif self.steps <= self.constant_steps:
            self.value = self.start_value
        else:
            self.value = self.end_value + 0.5 * (self.start_value - self.end_value) * (
                1 + math.cos((self.steps - self.constant_steps - 1) / (self.anneal_steps) * math.pi)
            )
        return self.value

    def __repr__(self):
        anneal_steps, constant_steps, start_value, end_value = (
            self.anneal_steps,
            self.constant_steps,
            self.start_value,
            self.end_value,
        )
        return f"CosineAnnealer({anneal_steps=}, {constant_steps=} {start_value=}, {end_value=})"
