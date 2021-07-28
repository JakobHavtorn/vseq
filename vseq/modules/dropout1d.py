import torch
import torch.nn as nn

class Dropout1d(nn.Module):

    def __init__(self, p):
        """
        Uses a fixed dropout mask across time steps. I.e., full channels are zeroed out.

        Args:
            p (float): Probability of an element to be zeroed
        """
        super().__init__()
        self.p = p

    def forward(self, input):
        """
        Args:
            input (Tensor): Should be 3D of shape BFT.

        Returns:
            Tensor: The input with zeroed out features.
        """

        if not self.training:
            return input

        assert input.ndim == 3, 'Expected a 3D tensor of shape TF (i.e., data from PackedSequence).'

        x = input
        F = x.size(1) # x is assumed
        do_mask = (torch.rand([1, F, 1], device=x.device) > self.p).to(torch.float) * (1.0 / (1.0 - self.p))
        x = x * do_mask
        return x

    def extra_repr(self):
        return str(self.p)