import abc

import torch
from torch import nn
from torch.distributions.utils import lazy_property

from utils.model_utils import init_linear_layer

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


class Distr(torch.distributions.Categorical, abc.ABC):
    @abc.abstractmethod
    def log_probs(self, actions: torch.LongTensor):
        raise NotImplementedError()


class CategoricalDistr(Distr):
    """A categorical distribution extending PyTorch's Categorical."""

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError()

    def cdf(self, value):
        raise Exception("CDF is not defined for categorical distributions.")

    def icdf(self, value):
        raise Exception("Inverse CDF is not defined for categorical distributions.")

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions: torch.LongTensor) -> torch.FloatTensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    @lazy_property
    def log_probs_tensor(self):
        return torch.log_softmax(self.logits, dim=-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):
    """A fixed normal distribution extending PyTorch's Normal."""

    def log_probs(self, actions: torch.LongTensor) -> torch.FloatTensor:
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean()


class FixedBernoulli(torch.distributions.Bernoulli):
    """A fixed Bernoulli distribution extending PyTorch's Bernoulli."""

    def log_probs(self, actions: torch.LongTensor) -> torch.FloatTensor:
        return super().log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class DiagGaussian(nn.Module):
    """A learned diagonal Gaussian distribution."""

    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        def init_(m):
            return init_linear_layer(
                m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
            )

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    """A learned Bernoulli distribution."""

    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        def init_(m):
            return init_linear_layer(
                m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
            )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


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

        return x + bias
