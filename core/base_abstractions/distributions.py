import abc

import torch
from torch import nn
from torch.distributions.utils import lazy_property

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


class Distr(torch.distributions.Categorical, abc.ABC):
    @abc.abstractmethod
    def log_probs(self, actions: torch.LongTensor):
        raise NotImplementedError()


class CategoricalDistr(Distr):
    """A categorical distribution extending PyTorch's Categorical."""

    def __init__(self, probs=None, logits=None, validate_args=None):
        super(CategoricalDistr, self).__init__(
            probs=probs, logits=logits, validate_args=validate_args
        )

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError()

    def cdf(self, value):
        raise Exception("CDF is not defined for categorical distributions.")

    def icdf(self, value):
        raise Exception("Inverse CDF is not defined for categorical distributions.")

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions: torch.LongTensor) -> torch.FloatTensor:
        return super().log_prob(actions.squeeze(-1)).unsqueeze(-1)

    @lazy_property
    def log_probs_tensor(self):
        return torch.log_softmax(self.logits, dim=-1)

    @lazy_property
    def probs_tensor(self):
        return torch.softmax(self.logits, dim=-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


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
