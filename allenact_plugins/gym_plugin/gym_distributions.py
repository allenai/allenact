import torch

from allenact.base_abstractions.distributions import Distr


class GaussianDistr(torch.distributions.Normal, Distr):
    """PyTorch's Normal distribution with a `mode` method."""

    def mode(self) -> torch.FloatTensor:
        return super().mean
