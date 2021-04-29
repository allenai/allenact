import abc
from typing import Any, Union, Callable, TypeVar, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.distributions.utils import lazy_property

from allenact.base_abstractions.sensor import ExpertActionSensor as Expert
from allenact.utils import spaces_utils as su


SampleType = TypeVar("SampleType")

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


class ConditionalDistr(Distr):
    # `action_group_name` is the identifier of the group of actions (OrderedDict) produced by this `ConditionalDistr`
    action_group_name: str

    def __init__(
        self,
        distr_conditioned_on_input_fn_or_instance: Union[Callable, Distr],
        action_group_name: str,
        *distr_conditioned_on_input_args,
        **distr_conditioned_on_input_kwargs,
    ):
        if isinstance(distr_conditioned_on_input_fn_or_instance, Distr):
            self.distr = distr_conditioned_on_input_fn_or_instance
        else:
            self.distr = None
            self.distr_conditioned_on_input_fn = (
                distr_conditioned_on_input_fn_or_instance
            )
            self.distr_conditioned_on_input_args = distr_conditioned_on_input_args
            self.distr_conditioned_on_input_kwargs = distr_conditioned_on_input_kwargs

        self.action_group_name = action_group_name

    def log_prob(self, actions):
        return self.distr.log_prob(actions)

    def entropy(self):
        return self.distr.entropy()

    def get_distr_conditioned_on_input(self, **ready_actions):
        if self.distr is None:
            assert all(
                key not in self.distr_conditioned_on_input_kwargs
                for key in ready_actions
            )
            self.distr = self.distr_conditioned_on_input_fn(
                *self.distr_conditioned_on_input_args,
                **self.distr_conditioned_on_input_kwargs,
                **ready_actions,
            )

    def sample(self, sample_shape=torch.Size(), **parent_actions):
        self.get_distr_conditioned_on_input(**parent_actions)
        act = self.distr.sample(sample_shape)
        return OrderedDict([(self.action_group_name, act)])

    def mode(self, **parent_actions):
        self.get_distr_conditioned_on_input(**parent_actions)
        act = self.distr.mode()
        return OrderedDict([(self.action_group_name, act)])


class SequentialDistr(Distr):
    def __init__(self, *conditional_distrs: ConditionalDistr):
        self.conditional_distrs = conditional_distrs

    def sample(self, sample_shape=torch.Size()):
        actions = OrderedDict()
        for cd in self.conditional_distrs:
            actions.update(cd.sample(sample_shape=sample_shape, **actions))
        return actions

    def mode(self):
        actions = OrderedDict()
        for cd in self.conditional_distrs:
            actions.update(cd.mode(**actions))
        return actions

    def conditional_entropy(self):
        sum = 0
        for cd in self.conditional_distrs:
            sum = sum + cd.entropy()
        return sum

    def entropy(self):
        raise NotImplementedError(
            "Please use `conditional_entropy` instead of `entropy` with SequentialDistr"
        )

    def log_prob(self, actions: Dict[str, Any]):
        assert len(actions) == len(
            self.conditional_distrs
        ), f"{len(self.conditional_distrs)} conditional distributions for {len(actions)} action groups"

        sum = 0
        for cd in self.conditional_distrs:
            cd.get_distr_conditioned_on_input(**actions)
            sum = sum + cd.log_prob(actions[cd.action_group_name])
        return sum


class TeacherForcingDistr(Distr):
    def __init__(
        self,
        distr,
        obs,
        action_space,
        active_samplers,
        approx_steps,
        teacher_forcing,
        tracking_info,
    ):
        self.distr = distr
        self.is_sequential = isinstance(self.distr, SequentialDistr)

        # action_space is a gym.spaces.Dict for SequentialDistr, or any gym.Space for other Distr
        self.action_space = action_space
        self.active_samplers = active_samplers
        self.approx_steps = approx_steps
        self.teacher_forcing = teacher_forcing
        self.tracking_info = tracking_info

        assert (
            "expert_action" in obs
        ), "When using teacher forcing, obs must contain an `expert_action` uuid"

        obs_space = Expert.flagged_space(
            self.action_space, use_dict_as_groups=self.is_sequential
        )
        self.expert = su.unflatten(obs_space, obs["expert_action"])

    def enforce(
        self,
        sample,
        action_space,
        teacher,
        teacher_force_info: Dict[str, Any],
        action_name: str = None,
    ):
        # expert_success is 0 if the expert action could not be computed and otherwise equals 1.
        tf_mask_shape = teacher[Expert.expert_success_label].shape
        expert_actions = teacher[Expert.action_label]
        expert_action_exists_mask = teacher[Expert.expert_success_label]

        actions = su.flatten(action_space, sample)

        assert (
            len(actions.shape) == 3
        ), f"Got flattened actions with shape {actions.shape} (it should be [1 x `samplers` x `flatdims`])"

        assert actions.shape[1] == self.active_samplers

        expert_actions = su.flatten(action_space, expert_actions)
        assert (
            expert_actions.shape == actions.shape
        ), f"expert actions shape {expert_actions.shape} doesn't match the model's {actions.shape}"

        teacher_forcing_mask = (
            torch.distributions.bernoulli.Bernoulli(
                torch.tensor(self.teacher_forcing(self.approx_steps))
            )
            .sample(tf_mask_shape)
            .long()
            .to(actions.device)
        ) * expert_action_exists_mask

        teacher_force_info[
            "teacher_ratio/sampled{}".format(
                f"_{action_name}" if action_name is not None else ""
            )
        ] = (teacher_forcing_mask.float().mean().item())

        extended_shape = teacher_forcing_mask.shape + (1,) * (
            len(actions.shape) - len(teacher_forcing_mask.shape)
        )

        actions = torch.where(
            teacher_forcing_mask.byte().view(extended_shape), expert_actions, actions
        )

        return su.unflatten(action_space, actions)

    def log_prob(self, actions: Any):
        return self.distr.log_prob(actions)

    def entropy(self):
        return self.distr.entropy()

    def conditional_entropy(self):
        return self.distr.conditional_entropy()

    def sample(self, sample_shape=torch.Size()):
        teacher_force_info = {
            "teacher_ratio/enforced": self.teacher_forcing(self.approx_steps),
        }

        if self.is_sequential:
            res = OrderedDict()
            for cd in self.distr.conditional_distrs:
                action_group_name = cd.action_group_name
                res[action_group_name] = self.enforce(
                    cd.sample(sample_shape, **res)[action_group_name],
                    self.action_space[action_group_name],
                    self.expert[action_group_name],
                    teacher_force_info,
                    action_group_name,
                )
        else:
            res = self.enforce(
                self.distr.sample(sample_shape),
                self.action_space,
                self.expert,
                teacher_force_info,
            )

        self.tracking_info["teacher"].append(
            ("teacher_package", teacher_force_info, self.active_samplers)
        )

        return res


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
