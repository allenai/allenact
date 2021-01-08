from typing import cast, Any

import abc

import torch
from torch import nn
from torch.distributions.utils import lazy_property

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


class Distr(abc.ABC):
    @abc.abstractmethod
    def log_probs(self, actions: Any):
        raise NotImplementedError()

    @abc.abstractmethod
    def entropy(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError()

    @abc.abstractmethod
    def mode(self):
        raise NotImplementedError()


class CategoricalDistr(torch.distributions.Categorical, Distr):
    """A categorical distribution extending PyTorch's Categorical."""

    def log_probs(self, actions: torch.LongTensor) -> torch.FloatTensor:
        # add an action dimension and squeeze step dimension
        return super().log_prob(actions).unsqueeze(-1).squeeze(0)

    def sample(self, sample_shape=torch.Size()):
        # squeeze the  step dimension
        return super().sample(sample_shape).squeeze(0)

    def mode(self):
        return self._param.argmax(dim=-1, keepdim=False)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError()

    def cdf(self, value):
        raise Exception("CDF is not defined for categorical distributions.")

    def icdf(self, value):
        raise Exception("Inverse CDF is not defined for categorical distributions.")

    @lazy_property
    def log_probs_tensor(self):
        return torch.log_softmax(self.logits, dim=-1)

    @lazy_property
    def probs_tensor(self):
        return torch.softmax(self.logits, dim=-1)


class AddBias(nn.Module):
    """Adding bias parameters to input values."""

    def __init__(self, bias: torch.FloatTensor):
        """Initializer.

        # Parameters

        bias : data to use as the initial values of the bias.
        """
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1), requires_grad=True)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:  # type: ignore
        """Adds the stored bias parameters to `x`."""
        assert x.dim() in [2, 4]

        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias  # type:ignore
