import torch


class STEFunction(torch.autograd.Function):
    """Abstract base class for Straight-Through Estimator"""
    @staticmethod
    def forward(ctx, x):
        raise NotImplementedError()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BinaryThresholdSTE(STEFunction):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()


class BernoulliSTE(STEFunction):
    """Given the probabilities p of outcome 1, draw binary outcomes {0, 1} with gradients defined by the STE"""
    @staticmethod
    def forward(ctx, p):
        return torch.bernoulli(p)
