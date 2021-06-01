import math


class Annealer:
    """Abstract base class for annealers"""

    value = None

    def step(self):
        raise NotImplementedError()


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

        if start_value != end_value:
            assert anneal_steps > 0, "With start_value != end_value, anneal_steps must be nonzero"

        self.anneal_steps = anneal_steps
        self.constant_steps = constant_steps
        self.start_value = start_value
        self.end_value = end_value
        self.steps = 0
        self.value = start_value

    def step(self):
        self.steps += 1

        if self.steps >= self.anneal_steps:
            self.value = self.end_value
        else:
            self.value = self.end_value + 0.5 * (self.start_value - self.end_value) * (
                1 + math.cos(self.steps / self.anneal_steps * math.pi)
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
