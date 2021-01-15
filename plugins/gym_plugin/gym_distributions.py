import torch

from core.base_abstractions.distributions import Distr


class GaussianDistr(torch.distributions.Normal, Distr):
    def mode(self) -> torch.FloatTensor:
        return super().mean
