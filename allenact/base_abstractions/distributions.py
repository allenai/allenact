import abc
from typing import Any, Union, Callable, TypeVar, Dict, Optional, cast
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.distributions.utils import lazy_property
import gym

from allenact.base_abstractions.sensor import AbstractExpertActionSensor as Expert
from allenact.utils import spaces_utils as su
from allenact.utils.misc_utils import all_unique

TeacherForcingAnnealingType = TypeVar("TeacherForcingAnnealingType")

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
    """Action distribution conditional which is conditioned on other information
       (i.e. part of a hierarchical distribution)

    # Attributes
    action_group_name : the identifier of the group of actions (`OrderedDict`) produced by this `ConditionalDistr`
    """

    action_group_name: str

    def __init__(
        self,
        distr_conditioned_on_input_fn_or_instance: Union[Callable, Distr],
        action_group_name: str,
        *distr_conditioned_on_input_args,
        **distr_conditioned_on_input_kwargs,
    ):
        self.distr: Optional[Distr] = None
        self.distr_conditioned_on_input_fn: Optional[Callable] = None
        self.distr_conditioned_on_input_args = distr_conditioned_on_input_args
        self.distr_conditioned_on_input_kwargs = distr_conditioned_on_input_kwargs

        if isinstance(distr_conditioned_on_input_fn_or_instance, Distr):
            self.distr = distr_conditioned_on_input_fn_or_instance
        else:
            self.distr_conditioned_on_input_fn = (
                distr_conditioned_on_input_fn_or_instance
            )

        self.action_group_name = action_group_name

    def log_prob(self, actions):
        return self.distr.log_prob(actions)

    def entropy(self):
        return self.distr.entropy()

    def condition_on_input(self, **ready_actions):
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

    def sample(self, sample_shape=torch.Size()) -> OrderedDict:
        return OrderedDict([(self.action_group_name, self.distr.sample(sample_shape))])

    def mode(self) -> OrderedDict:
        return OrderedDict([(self.action_group_name, self.distr.mode())])


class SequentialDistr(Distr):
    def __init__(self, *conditional_distrs: ConditionalDistr):
        action_group_names = [cd.action_group_name for cd in conditional_distrs]
        assert all_unique(
            action_group_names
        ), f"All conditional distribution `action_group_name`, must be unique, given names {action_group_names}"
        self.conditional_distrs = conditional_distrs

    def sample(self, sample_shape=torch.Size()):
        actions = OrderedDict()
        for cd in self.conditional_distrs:
            cd.condition_on_input(**actions)
            actions.update(cd.sample(sample_shape=sample_shape))
        return actions

    def mode(self):
        actions = OrderedDict()
        for cd in self.conditional_distrs:
            cd.condition_on_input(**actions)
            actions.update(cd.mode())
        return actions

    def conditional_entropy(self):
        sum = 0
        for cd in self.conditional_distrs:
            sum = sum + cd.entropy()
        return sum

    def entropy(self):
        raise NotImplementedError(
            "Please use 'conditional_entropy' instead of 'entropy' as the `entropy_method_name` "
            "parameter in your loss when using `SequentialDistr`."
        )

    def log_prob(self, actions: Dict[str, Any]):
        assert len(actions) == len(
            self.conditional_distrs
        ), f"{len(self.conditional_distrs)} conditional distributions for {len(actions)} action groups"

        sum = 0
        for cd in self.conditional_distrs:
            cd.condition_on_input(**actions)
            sum = sum + cd.log_prob(actions[cd.action_group_name])
        return sum


class TeacherForcingDistr(Distr):
    def __init__(
        self,
        distr: Distr,
        obs: Dict[str, Any],
        action_space: gym.spaces.Space,
        num_active_samplers: int,
        approx_steps: int,
        teacher_forcing: TeacherForcingAnnealingType,
        tracking_info: Dict[str, Any],
    ):
        self.distr = distr
        self.is_sequential = isinstance(self.distr, SequentialDistr)

        # action_space is a gym.spaces.Dict for SequentialDistr, or any gym.Space for other Distr
        self.action_space = action_space
        self.num_active_samplers = num_active_samplers
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
        sample: Any,
        action_space: gym.spaces.Space,
        teacher: OrderedDict,
        teacher_force_info: Dict[str, Any],
        action_name: str = None,
    ):
        actions = su.flatten(action_space, sample)

        assert (
            len(actions.shape) == 3
        ), f"Got flattened actions with shape {actions.shape} (it should be [1 x `samplers` x `flatdims`])"

        assert actions.shape[1] == self.num_active_samplers

        expert_actions = su.flatten(action_space, teacher[Expert.ACTION_POLICY_LABEL])
        assert (
            expert_actions.shape == actions.shape
        ), f"expert actions shape {expert_actions.shape} doesn't match the model's {actions.shape}"

        # expert_success is 0 if the expert action could not be computed and otherwise equals 1.
        expert_action_exists_mask = teacher[Expert.EXPERT_SUCCESS_LABEL]

        teacher_forcing_mask = (
            torch.distributions.bernoulli.Bernoulli(
                torch.tensor(self.teacher_forcing(self.approx_steps))
            )
            .sample(expert_action_exists_mask.shape)
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
            for cd in cast(SequentialDistr, self.distr).conditional_distrs:
                cd.condition_on_input(**res)
                action_group_name = cd.action_group_name
                res[action_group_name] = self.enforce(
                    cd.sample(sample_shape)[action_group_name],
                    cast(gym.spaces.Dict, self.action_space)[action_group_name],
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
            ("teacher_package", teacher_force_info, self.num_active_samplers)
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
