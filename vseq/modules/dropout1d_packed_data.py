import torch
import torch.nn as nn

class Dropout1dPackedData(nn.Module):

    def __init__(self, p):
        """
        Uses a fixed dropout mask across time steps.

        Args:
            p (float): Probability of an element to be zeroed
        """
        super().__init__()
        self.p = p

    def forward(self, input):
        """
        Args:
            input (Tensor): Should be 2D of shape TF.

        Returns:
            Tensor: The input with zeroed out features along the temporal dimension.
        """

        if not self.training:
            return input

        assert input.ndim == 2, 'Expected a 2D tensor of shape TF (i.e., data from PackedSequence).'

        x = input
        F = x.size(1) # x is assumed
        do_mask = (torch.rand([1, F], device=x.device) > self.p).to(torch.float) * (1.0 / (1.0 - self.p))
        x = x * do_mask
        return x

    def extra_repr(self):
        return str(self.p)