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
