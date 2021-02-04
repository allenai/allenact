import abc
from typing import Any

import torch
from torch import nn
from torch.distributions.utils import lazy_property

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


class Distr(abc.ABC):
    @abc.abstractmethod
    def log_prob(self, actions: Any):
        """Return the log probability/ies of the provided action/s."""
        raise NotImplementedError()

    @abc.abstractmethod
    def entropy(self):
        """Return the entropy or entropies."""
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self, sample_shape=torch.Size()):
        """Sample actions."""
        raise NotImplementedError()

    def mode(self):
        """If available, return the action(s) with highest probability.

        It will only be called if using deterministic agents.
        """
        raise NotImplementedError()


class CategoricalDistr(torch.distributions.Categorical, Distr):
    """A categorical distribution extending PyTorch's Categorical.

    probs or logits are assumed to be passed with step and sampler
    dimensions as in: [step, samplers, ...]
    """

    def mode(self):
        return self._param.argmax(dim=-1, keepdim=False)  # match sample()'s shape

    def log_prob(self, value: torch.Tensor):
        if value.shape == self.logits.shape[:-1]:
            return super(CategoricalDistr, self).log_prob(value=value)
        elif value.shape == self.logits.shape[:-1] + (1,):
            return (
                super(CategoricalDistr, self)
                .log_prob(value=value.squeeze(-1))
                .unsqueeze(-1)
            )
        else:
            raise NotImplementedError(
                "Broadcasting in categorical distribution is disabled as it often leads"
                f" to unexpected results. We have that `value.shape == {value.shape}` but"
                f" expected a shape of "
                f" `self.logits.shape[:-1] == {self.logits.shape[:-1]}` or"
                f" `self.logits.shape[:-1] + (1,) == {self.logits.shape[:-1] + (1,)}`"
            )

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
