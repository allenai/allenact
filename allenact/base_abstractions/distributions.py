import abc
from typing import Any, Tuple, Optional, Sequence, Union, Callable, TypeVar, Dict, List
from collections import OrderedDict

import torch
from torch import nn
from torch.distributions.utils import lazy_property
import networkx as nx
import gym

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


class CondDistr(Distr):
    produces: Sequence[str]
    given: Optional[Sequence[str]] = None
    action_space: gym.spaces.Space

    def __init__(
        self,
        action_space: gym.spaces.Space,
        get_subpolicy: Callable,
        produces: Sequence[str],
        given: Optional[Sequence[str]] = None,
        *sub_args,
        **kwsub_args,
    ):
        self.action_space = action_space
        self.get_subpolicy_fn = get_subpolicy
        self.subpolicy = None
        self.sub_args = sub_args
        self.kwsub_args = kwsub_args

        self.produces = produces
        self.given = given

    def log_prob(self, actions):
        assert self.subpolicy is not None
        return self.subpolicy.log_prob(actions)

    def entropy(self):
        assert self.subpolicy is not None
        return self.subpolicy.entropy()

    def get_subpolicy(self, **kwconds):
        assert all([key not in self.kwsub_args for key in kwconds])
        self.subpolicy = self.get_subpolicy_fn(
            *self.sub_args, **self.kwsub_args, **kwconds
        )

    def sample(self, sample_shape=torch.Size(), **kwconds):
        self.get_subpolicy(**kwconds)
        return self.subpolicy.sample(sample_shape)

    def mode(self, **kwconds):
        self.get_subpolicy(**kwconds)
        return self.subpolicy.mode()


class DirectedGraphicalModel(Distr):
    def __init__(self, cond_distrs: Sequence[CondDistr]):
        self.partials: List[CondDistr]
        self.deps: Dict[Tuple[str], List[Tuple[str]]]
        self.partials, self.deps = DirectedGraphicalModel.make_ordered_partials(
            cond_distrs
        )
        assert len(self.partials) == len(
            cond_distrs
        ), f"Inconsistent number of used ({len(self.partials)}) and given ({len(cond_distrs)}) partial distributions"
        self.action_spaces = [partial.action_space for partial in self.partials]

    @staticmethod
    def collapse_deps(
        varnames: Sequence[str], who_produces: Dict[str, Tuple[str]]
    ) -> List[Tuple[str, ...]]:
        """Generates a list of produced partial distributions required by the list of varnames.
        """
        current_partials = set()
        for d in varnames:
            current_partials.add(who_produces[d])
        return list(current_partials)

    @staticmethod
    def make_ordered_partials(
        cond_distrs: Sequence[CondDistr],
    ) -> Tuple[List[CondDistr], Dict[Tuple[str], List[Tuple[str]]]]:
        """Generates a computation order given a list of `CondDistr`.
           It also provides a dict with produced partial distributions required by the list of produced vars.
        """
        producer: Dict[Tuple[str, ...], CondDistr] = {
            tuple(d.produces): d for d in cond_distrs
        }
        who_produces: Dict[str, Tuple[str]] = {}
        deps: Dict[Tuple[str], List[Union[str, Tuple[str]]]] = {
            tuple(d.produces): list(d.given) for d in cond_distrs if d.given is not None
        }
        for partial in cond_distrs:
            assert (
                partial.produces is not None
            ), "produces must be provided for cond_distrs"
            for varname in partial.produces:
                assert (
                    varname not in who_produces
                ), "'{}' is a duplicated produced variable".format(varname)
                who_produces[varname] = tuple(partial.produces)

        g = nx.DiGraph()
        for k in deps:
            deps[k] = DirectedGraphicalModel.collapse_deps(
                deps[k], who_produces
            )  # List[str] -> List[Tuple[str]]
            for j in deps[k]:
                g.add_edge(j, k)

        assert nx.is_directed_acyclic_graph(
            g
        ), "partial distributions do not form a direct acyclic graph"

        # ensure dependencies are precomputed
        return [producer[n] for n in nx.dfs_postorder_nodes(g)][::-1], deps

    def entropy(self):
        sum = 0
        for partial in self.partials:
            sum = sum + partial.entropy()
        return sum

    def log_prob(self, actions: Sequence[Dict[str, Any]]):
        assert len(actions) == len(
            self.partials
        ), f"{len(self.partials)} partial distributions for {len(actions)} partial actions"
        sum = 0
        prods = [tuple(partial.produces) for partial in self.partials]
        for acts, partial in zip(actions, self.partials):
            if partial.subpolicy is None:
                deps = (
                    [k for k in self.deps[tuple(partial.produces)]]
                    if partial.given is not None
                    else {}
                )
                if len(deps) > 0:
                    idxs = [prods.index(dep) for dep in deps]
                    deps = {
                        name: actions[idx] for k, idx in zip(deps, idxs) for name in k
                    }
                partial.get_subpolicy(**deps)
            sum = sum + partial.log_prob(acts)
        return sum

    def sample(self, sample_shape=torch.Size()) -> Tuple:
        res = OrderedDict()
        for partial in self.partials:
            deps = (
                {name: res[k] for k in self.deps[tuple(partial.produces)] for name in k}
                if partial.given is not None
                else {}
            )
            res[tuple(partial.produces)] = partial.sample(sample_shape, **deps)
        return tuple(res.values())

    def mode(self) -> Tuple:
        res = OrderedDict()
        for partial in self.partials:
            deps = (
                {name: res[k] for k in self.deps[tuple(partial.produces)] for name in k}
                if partial.given is not None
                else {}
            )
            res[tuple(partial.produces)] = partial.mode(**deps)
        return tuple(res.values())


class TeacherForcingDistr(Distr):
    def __init__(self, distr, obs, engine):
        self.engine = engine  # access to action space, training steps, ...
        self.distr = distr  # graphical model or plain Distr
        self.use_dag = isinstance(self.distr, DirectedGraphicalModel)

        assert (
            "expert_action" in obs
        ), "When using teacher forcing, obs must contain an `expert_action` uuid"

        # obs["expert_action] can be Tuple (if using DirectedGraphicalModel/group_spaces), or OrderedDict:
        obs_space = self.expert_action_space()
        self.obs = su.unflatten(obs_space, obs["expert_action"])

        if self.use_dag:
            self.group_spaces = tuple(self.distr.action_spaces)

            assert isinstance(
                self.obs, Tuple
            ), f"received {type(self.obs)} expert observation when using DirectedGraphicalModel (expected tuple)"

            assert len(self.distr.partials) == len(
                self.obs
            ), f"{len(self.distr.partials)} partial `CondDistr`s in distr for {len(self.obs)} expert groups"
        else:
            self.group_spaces = (self.engine.actor_critic.action_space,)

            assert isinstance(
                self.obs, OrderedDict
            ), f"received {type(self.obs)} expert observation when NOT using groups (expected OrderedDict)"

        self.num_active_samplers = None

    def expert_action_space(self):
        obs_space = self.engine.vector_tasks.action_spaces[0]

        if not self.use_dag:
            return Expert.flagged_group_space(obs_space)
        else:
            return gym.spaces.Tuple(
                [Expert.flagged_group_space(group_space) for group_space in obs_space],
            )

    @lazy_property
    def approx_steps(self):
        if self.engine.is_distributed:
            # the actual number of steps gets synchronized after each rollout
            return (
                self.engine.step_count - self.engine.former_steps
            ) * self.engine.num_workers + self.engine.former_steps
        else:
            return self.engine.step_count  # this is actually accurate

    def enforce(
        self, sample, action_space, teacher, teacher_force_info: Dict[str, Any], step=-1
    ):
        # expert_success is 0 if the expert action could not be computed and otherwise equals 1.
        tf_mask_shape = teacher[Expert.expert_success_label].shape
        expert_actions = teacher[Expert.action_label]
        expert_action_exists_mask = teacher[Expert.expert_success_label]

        actions = su.flatten(action_space, sample)

        assert (
            len(actions.shape) == 3
        ), f"got flattened actions with shape {actions.shape}"

        if self.num_active_samplers is None:
            self.num_active_samplers = actions.shape[1]
        else:
            assert actions.shape[1] == self.num_active_samplers

        expert_actions = su.flatten(action_space, expert_actions)
        assert (
            expert_actions.shape == actions.shape
        ), f"expert actions shape {expert_actions.shape} doesn't match the model's {actions.shape}"

        teacher_forcing_mask = (
            torch.distributions.bernoulli.Bernoulli(
                torch.tensor(
                    self.engine.training_pipeline.current_stage.teacher_forcing(
                        self.approx_steps
                    )
                )
            )
            .sample(tf_mask_shape)
            .long()
            .to(actions.device)
        ) * expert_action_exists_mask

        teacher_force_info[
            "teacher_ratio/sampled{}".format(step if step >= 0 else "")
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

    def sample(self, sample_shape=torch.Size()):
        teacher_force_info = {
            "teacher_ratio/enforced": self.engine.training_pipeline.current_stage.teacher_forcing(
                self.approx_steps
            ),
        }

        if self.use_dag:
            res = OrderedDict()
            for step, partial, aspace, expert in zip(
                range(len(self.distr.partials)),
                self.distr.partials,
                self.group_spaces,
                self.obs,
            ):
                deps = (
                    {
                        name: res[k]
                        for k in self.distr.deps[tuple(partial.produces)]
                        for name in k
                    }
                    if partial.given is not None
                    else {}
                )
                res[tuple(partial.produces)] = self.enforce(
                    partial.sample(sample_shape, **deps),
                    aspace,
                    expert,
                    teacher_force_info,
                    step,
                )
            res = tuple(res.values())
        else:
            res = self.enforce(
                self.distr.sample(sample_shape),
                self.group_spaces[0],
                self.obs,
                teacher_force_info,
            )

        self.engine.tracking_info["teacher"].append(
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
