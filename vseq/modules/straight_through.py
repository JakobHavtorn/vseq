import torch


class STEFunction(torch.autograd.Function):
    """Abstract base class for Straight-Through Estimator"""
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        raise NotImplementedError()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class BinaryThresholdSTEFunction(STEFunction):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return (x > 0.5).float()


class BernoulliSTEFunction(STEFunction):
    """Given the probabilities p of outcome 1, draw binary outcomes {0, 1} with gradients defined by the STE"""
    @staticmethod
    def forward(ctx, p: torch.Tensor):
        return torch.bernoulli(p)


class BernoulliSTE(torch.jit.ScriptModule):
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()

    @torch.jit.script_method
    def forward(self, p: torch.Tensor):
        return p + (torch.bernoulli(p) - p).detach()


class BinaryThresholdSTE(torch.jit.ScriptModule):
    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return x + ((x > self.threshold).float() - x).detach()


if __name__ == "__main__":
    import timeit
    import numpy as np


    class BernoulliSTE_nojit(torch.nn.Module):
        def __init__(self, threshold: float = 0.5) -> None:
            super().__init__()

        def forward(self, p: torch.Tensor):
            return p + (torch.bernoulli(p) - p).detach()


    class BinaryThresholdSTE_nojit(torch.nn.Module):
        def __init__(self, threshold: float = 0.5) -> None:
            super().__init__()
            self.threshold = threshold

        def forward(self, x: torch.Tensor):
            return x + ((x > self.threshold).float() - x).detach()


    def time_function(eval_string, **namespace):
        timer = timeit.Timer(eval_string, globals=namespace)
        number, time_taken = timer.autorange()
        timings = timer.repeat(repeat=20, number=number)
        timings = [t / number for t in timings]
        print(f"{eval_string:10s}: {number=:d}, {min(timings):.3e} +- {np.std(timings):.3e} s")
        return timings, number


    for device in ['cpu', 'cuda']:
        print(f"\nTimings for {device}:\n")
        binary_ste_jit = BinaryThresholdSTE().to(device)
        bernoulli_ste_jit = BernoulliSTE().to(device)

        binary_ste = BinaryThresholdSTE_nojit().to(device)
        bernoulli_ste = BernoulliSTE_nojit().to(device)

        p = torch.rand(256, 100, 100).to(device)

        timings, number = time_function("binary_ste_jit(p)", binary_ste_jit=binary_ste_jit, p=p)
        timings, number = time_function("binary_ste(p)", binary_ste=binary_ste, p=p)
        
        timings, number = time_function("bernoulli_ste_jit(p)", bernoulli_ste_jit=bernoulli_ste_jit, p=p)
        timings, number = time_function("bernoulli_ste(p)", bernoulli_ste=bernoulli_ste, p=p)
