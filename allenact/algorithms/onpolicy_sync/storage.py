# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import random
from typing import (
    Union,
    List,
    Dict,
    Tuple,
    Sequence,
    cast,
    Optional,
    Callable,
    Any,
    Generator,
)

import gym
import numpy as np
import torch
from torch import Tensor

import allenact.utils.spaces_utils as su
from allenact.algorithms.onpolicy_sync.policy import (
    FullMemorySpecType,
    ObservationType,
    ActionType,
)
from allenact.base_abstractions.misc import Memory


class ExperienceStorage(abc.ABC):
    @abc.abstractmethod
    def initialize(self, *, observations: ObservationType, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def add(
        self,
        observations: ObservationType,
        memory: Optional[Memory],
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ):
        """
        # Parameters
        observations : Observations after taking `actions`
        memory: Memory after having observed the last set of observations.
        actions: Actions taken to reach the current state, i.e. taking these actions has led to a new state with
            new `observations`.
        action_log_probs : Log probs of `actions`
        value_preds : Value predictions corresponding to the last observations
            (i.e. the states before taking `actions`).
        rewards : Rewards from taking `actions` in the last set of states.
        masks : Masks corresponding to the current states, having 0 entries where `observations` correspond to
            observations from the beginning of a new episode.
        """
        raise NotImplementedError

    def before_updates(self, **kwargs):
        pass

    def after_updates(self, **kwargs) -> int:
        pass

    @abc.abstractmethod
    def to(self, device: torch.device):
        pass

    @abc.abstractmethod
    def set_partition(self, index: int, num_parts: int):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def total_experiences(self) -> int:
        raise NotImplementedError


class RolloutStorage(ExperienceStorage, abc.ABC):
    # noinspection PyMethodOverriding
    @abc.abstractmethod
    def initialize(
        self,
        *,
        observations: ObservationType,
        num_samplers: int,
        recurrent_memory_specification: FullMemorySpecType,
        action_space: gym.Space,
        **kwargs,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def agent_input_for_next_step(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def sampler_select(self, keep_list: Sequence[int]):
        raise NotImplementedError


class StreamingStorageMixin(abc.ABC):
    @abc.abstractmethod
    def next_batch(self) -> Dict[str, Any]:
        raise NotImplementedError

    def reset_stream(self):
        raise NotImplementedError

    @abc.abstractmethod
    def empty(self) -> bool:
        raise NotImplementedError


class MiniBatchStorageMixin(abc.ABC):
    @abc.abstractmethod
    def batched_experience_generator(
        self, num_mini_batch: int,
    ) -> Generator[Dict[str, Any], None, None]:
        raise NotImplementedError


class RolloutBlockStorage(RolloutStorage, MiniBatchStorageMixin):
    """Class for storing rollout information for RL trainers."""

    FLATTEN_SEPARATOR: str = "._AUTOFLATTEN_."

    def __init__(self, init_size: int = 50):
        self.full_size = init_size

        self.flattened_to_unflattened: Dict[str, Dict[str, List[str]]] = {
            "memory": dict(),
            "observations": dict(),
        }
        self.unflattened_to_flattened: Dict[str, Dict[Tuple[str, ...], str]] = {
            "memory": dict(),
            "observations": dict(),
        }

        self.dim_names = ["step", "sampler", None]

        self.memory_specification: Optional[FullMemorySpecType] = None
        self.action_space: Optional[gym.Space] = None
        self.memory_first_last: Optional[Memory] = None
        self._observations_full: Memory = Memory()

        self._value_preds_full: Optional[torch.Tensor] = None
        self._returns_full: Optional[torch.Tensor] = None
        self._rewards_full: Optional[torch.Tensor] = None
        self._action_log_probs_full: Optional[torch.Tensor] = None

        self.step = 0
        self._total_steps = 0
        self._before_update_called = False
        self.device = torch.device("cpu")

        # self._advantages and self._normalized_advantages are only computed
        # when `before_updates` is called
        self._advantages: Optional[torch.Tensor] = None
        self._normalized_advantages: Optional[torch.Tensor] = None

        self._masks_full: Optional[torch.Tensor] = None
        self._actions_full: Optional[torch.Tensor] = None
        self._prev_actions_full: Optional[torch.Tensor] = None

    def initialize(
        self,
        observations: ObservationType,
        num_samplers: int,
        recurrent_memory_specification: FullMemorySpecType,
        action_space: gym.Space,
        **kwargs,
    ):
        if self.memory_specification is None:
            self.memory_specification = recurrent_memory_specification
            self.action_space = action_space

            self.memory_first_last: Memory = self.create_memory(
                spec=self.memory_specification, num_samplers=num_samplers,
            ).to(self.device)
            for key in self.memory_specification:
                self.flattened_to_unflattened["memory"][key] = [key]
                self.unflattened_to_flattened["memory"][(key,)] = key

            self._masks_full = torch.zeros(
                self.full_size + 1, num_samplers, 1, device=self.device
            )
            action_flat_dim = su.flatdim(self.action_space)
            self._actions_full = torch.zeros(
                self.full_size, num_samplers, action_flat_dim, device=self.device
            )
            self._prev_actions_full = torch.zeros(
                self.full_size + 1, num_samplers, action_flat_dim, device=self.device
            )

        assert self.step == 0
        self.insert_observations(observations=observations, time_step=0)
        self.prev_actions[0].zero_()  # Have to zero previous actions
        self.masks[0].zero_()  # Have to zero masks

    @property
    def total_experiences(self) -> int:
        return self._total_steps

    @total_experiences.setter
    def total_experiences(self, value: int):
        self._total_steps = value

    def set_partition(self, index: int, num_parts: int):
        pass

    @property
    def value_preds(self) -> torch.Tensor:
        return self._value_preds_full[: self.step + 1]

    @property
    def rewards(self) -> torch.Tensor:
        return self._rewards_full[: self.step]

    @property
    def returns(self) -> torch.Tensor:
        return self._returns_full[: self.step + 1]

    @property
    def action_log_probs(self) -> torch.Tensor:
        return self._action_log_probs_full[: self.step]

    @property
    def actions(self) -> torch.Tensor:
        return self._actions_full[: self.step]

    @property
    def prev_actions(self) -> torch.Tensor:
        return self._prev_actions_full[: self.step + 1]

    @property
    def masks(self) -> torch.Tensor:
        return self._masks_full[: self.step + 1]

    @property
    def observations(self) -> Memory:
        return self._observations_full.slice(dim=0, start=0, stop=self.step + 1)

    @staticmethod
    def create_memory(spec: Optional[FullMemorySpecType], num_samplers: int,) -> Memory:
        if spec is None:
            return Memory()

        memory = Memory()
        for key in spec:
            dims_template, dtype = spec[key]

            dim_names = ["step"] + [d[0] for d in dims_template]
            sampler_dim = dim_names.index("sampler")

            all_dims = [2] + [d[1] for d in dims_template]
            all_dims[sampler_dim] = num_samplers

            memory.check_append(
                key=key,
                tensor=torch.zeros(*all_dims, dtype=dtype),
                sampler_dim=sampler_dim,
            )

        return memory

    def to(self, device: torch.device):
        for key in [
            "_observations_full",
            "memory_first_last",
            "_actions_full",
            "_prev_actions_full",
            "_masks_full",
            "_rewards_full",
            "_value_preds_full",
            "_returns_full",
            "_action_log_probs_full",
        ]:
            val = getattr(self, key)
            if val is not None:
                setattr(self, key, val.to(device))

        self.device = device

    def insert_observations(
        self, observations: ObservationType, time_step: int,
    ):
        self.insert_tensors(
            storage=self._observations_full,
            storage_name="observations",
            unflattened=observations,
            time_step=time_step,
        )

    def insert_memory(
        self, memory: Optional[Memory], time_step: int,
    ):
        if memory is None:
            assert len(self.memory_first_last) == 0
            return

        # `min(time_step, 1)` as we only store the first and last memories:
        #  * first memory is used for loss computation when the agent model has to compute
        #    all its outputs again given the full batch.
        #  * last memory ised used by the agent when collecting rollouts
        self.insert_tensors(
            storage=self.memory_first_last,
            storage_name="memory",
            unflattened=memory,
            time_step=min(time_step, 1),
        )

    def insert_tensors(
        self,
        storage: Memory,
        storage_name: str,
        unflattened: Union[ObservationType, Memory],
        prefix: str = "",
        path: Sequence[str] = (),
        time_step: int = 0,
    ):
        path = list(path)

        for name in unflattened:
            current_data = unflattened[name]

            if isinstance(current_data, Dict):
                self.insert_tensors(
                    storage=storage,
                    storage_name=storage_name,
                    unflattened=cast(ObservationType, current_data),
                    prefix=prefix + name + self.FLATTEN_SEPARATOR,
                    path=path + [name],
                    time_step=time_step,
                )
                continue

            sampler_dim = self.dim_names.index("sampler")
            if isinstance(current_data, tuple):
                sampler_dim = current_data[1]
                current_data = current_data[0]

            flatten_name = prefix + name
            if flatten_name not in storage:
                assert storage_name == "observations"
                storage[flatten_name] = (
                    torch.zeros_like(current_data)  # type:ignore
                    .repeat(
                        self.full_size + 1,  # required for observations (and memory)
                        *(1 for _ in range(len(current_data.shape))),
                    )
                    .to(self.device),
                    sampler_dim,
                )

                assert (
                    flatten_name not in self.flattened_to_unflattened[storage_name]
                ), f"new flattened name {flatten_name} already existing in flattened spaces[{storage_name}]"
                self.flattened_to_unflattened[storage_name][flatten_name] = path + [
                    name
                ]
                self.unflattened_to_flattened[storage_name][
                    tuple(path + [name])
                ] = flatten_name

            if storage_name == "observations":
                # current_data has a step dimension
                assert time_step >= 0
                storage[flatten_name][0][time_step : time_step + 1].copy_(current_data)
            elif storage_name == "memory":
                # current_data does not have a step dimension
                storage[flatten_name][0][time_step].copy_(current_data)
            else:
                raise NotImplementedError

    def create_tensor_storage(
        self, num_steps: int, template: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([torch.zeros_like(template).to(self.device)] * num_steps)

    def _double_storage_size(self):
        def pad_tensor_with_zeros(old_t: Optional[torch.Tensor]):
            if old_t is None:
                return None

            assert old_t.shape[0] in [self.full_size, self.full_size + 1]
            padded_t = torch.zeros(
                old_t.shape[0] + self.full_size,
                *old_t.shape[1:],
                dtype=old_t.dtype,
                device=old_t.device,
            )
            padded_t[: old_t.shape[0]] = old_t
            return padded_t

        for key in list(self._observations_full.keys()):
            obs_tensor, sampler_dim = self._observations_full[key]
            self._observations_full[key] = (
                pad_tensor_with_zeros(obs_tensor),
                sampler_dim,
            )

        self._actions_full = pad_tensor_with_zeros(self._actions_full)
        self._prev_actions_full = pad_tensor_with_zeros(self._prev_actions_full)
        self._masks_full = pad_tensor_with_zeros(self._masks_full)

        self._rewards_full = pad_tensor_with_zeros(self._rewards_full)
        self._value_preds_full = pad_tensor_with_zeros(self._value_preds_full)
        self._returns_full = pad_tensor_with_zeros(self._returns_full)
        self._action_log_probs_full = pad_tensor_with_zeros(self._action_log_probs_full)

        self.full_size *= 2

    def add(
        self,
        observations: ObservationType,
        memory: Optional[Memory],
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ):
        """See `ExperienceStorage.add` documentation."""
        assert (
            len(masks.shape) == 2 and masks.shape[1] == 1
        ), f"Can only add a single step worth of data at a time (mask shape = {masks.shape})."

        self.total_experiences += masks.shape[0]

        if self.step == self.full_size:
            self._double_storage_size()
        elif self.step > self.full_size:
            raise RuntimeError

        self.insert_observations(observations, time_step=self.step + 1)
        self.insert_memory(memory, time_step=self.step + 1)

        assert actions.shape == self._actions_full.shape[1:]

        self._actions_full[self.step].copy_(actions)  # type:ignore
        self._prev_actions_full[self.step + 1].copy_(actions)  # type:ignore
        self._masks_full[self.step + 1].copy_(masks)  # type:ignore

        if self._rewards_full is None:
            # We delay the instantiation of storage for `rewards`, `value_preds`, `action_log_probs` and `returns`
            # as we do not, a priori, know what shape these will be. For instance, if we are in a multi-agent setting
            # then there may be many rewards (one for each agent).
            self._rewards_full = self.create_tensor_storage(
                self.full_size, rewards.unsqueeze(0)
            )  # add step

            value_returns_template = value_preds.unsqueeze(0)  # add step
            self._value_preds_full = self.create_tensor_storage(
                self.full_size + 1, value_returns_template
            )
            self._returns_full = self.create_tensor_storage(
                self.full_size + 1, value_returns_template
            )

            self._action_log_probs_full = self.create_tensor_storage(
                self.full_size, action_log_probs.unsqueeze(0)
            )

        self._value_preds_full[self.step].copy_(value_preds)  # type:ignore
        self._rewards_full[self.step].copy_(rewards)  # type:ignore
        self._action_log_probs_full[self.step].copy_(  # type:ignore
            action_log_probs
        )

        self.step += 1
        self._before_update_called = False

        # We set the below to be None just for extra safety.
        self._advantages = None
        self._normalized_advantages = None

    def sampler_select(self, keep_list: Sequence[int]):
        keep_list = list(keep_list)
        if self._actions_full.shape[1] == len(keep_list):  # samplers dim
            return  # we are keeping everything, no need to copy

        self._observations_full = self._observations_full.sampler_select(keep_list)
        self.memory_first_last = self.memory_first_last.sampler_select(keep_list)
        self._actions_full = self._actions_full[:, keep_list]
        self._prev_actions_full = self._prev_actions_full[:, keep_list]
        self._action_log_probs_full = self._action_log_probs_full[:, keep_list]
        self._masks_full = self._masks_full[:, keep_list]

        if self._rewards_full is not None:
            self._value_preds_full = self._value_preds_full[:, keep_list]
            self._rewards_full = self._rewards_full[:, keep_list]
            self._returns_full = self._returns_full[:, keep_list]

    def before_updates(
        self,
        next_value: torch.Tensor,
        use_gae: bool,
        gamma: float,
        tau: float,
        adv_stats_callback: Callable[[torch.Tensor], Dict[str, Tensor]],
        **kwargs,
    ):
        assert len(kwargs) == 0
        self.compute_returns(
            next_value=next_value, use_gae=use_gae, gamma=gamma, tau=tau,
        )

        self._advantages = self.returns[:-1] - self.value_preds[:-1]

        adv_stats = adv_stats_callback(self._advantages)
        self._normalized_advantages = (self._advantages - adv_stats["mean"]) / (
            adv_stats["std"] + 1e-5
        )

        self._before_update_called = True

    def after_updates(self, **kwargs):
        assert len(kwargs) == 0

        for storage in [self.observations, self.memory_first_last]:
            for key in storage:
                storage[key][0][0].copy_(storage[key][0][-1])

        self.masks[0].copy_(self.masks[-1])
        self.prev_actions[0].copy_(self.prev_actions[-1])

        self._before_update_called = False
        self._advantages = None
        self._normalized_advantages = None
        self.step = 0

    @staticmethod
    def _extend_tensor_with_ones(stored_tensor: torch.Tensor, desired_num_dims: int):
        # Ensure broadcast to all flattened dimensions
        extended_shape = stored_tensor.shape + (1,) * (
            desired_num_dims - len(stored_tensor.shape)
        )
        return stored_tensor.view(*extended_shape)

    def compute_returns(
        self, next_value: torch.Tensor, use_gae: bool, gamma: float, tau: float
    ):
        extended_mask = self._extend_tensor_with_ones(
            self.masks, desired_num_dims=len(self.value_preds.shape)
        )
        extended_rewards = self._extend_tensor_with_ones(
            self.rewards, desired_num_dims=len(self.value_preds.shape)
        )

        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(extended_rewards.shape[0])):
                delta = (
                    extended_rewards[step]
                    + gamma * self.value_preds[step + 1] * extended_mask[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * extended_mask[step + 1] * gae  # type:ignore
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(extended_rewards.shape[0])):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * extended_mask[step + 1]
                    + extended_rewards[step]
                )

    def batched_experience_generator(
        self, num_mini_batch: int,
    ):
        assert self._before_update_called, (
            "self._before_update_called() must be called before"
            " attempting to generated batched rollouts."
        )
        num_samplers = self.rewards.shape[1]
        assert num_samplers >= num_mini_batch, (
            f"The number of task samplers ({num_samplers}) "
            f"must be greater than or equal to the number of "
            f"mini batches ({num_mini_batch})."
        )

        inds = np.round(
            np.linspace(0, num_samplers, num_mini_batch + 1, endpoint=True)
        ).astype(np.int32)
        pairs = list(zip(inds[:-1], inds[1:]))
        random.shuffle(pairs)

        for start_ind, end_ind in pairs:
            cur_samplers = list(range(start_ind, end_ind))

            memory_batch = self.memory_first_last.step_squeeze(0).sampler_select(
                cur_samplers
            )
            observations_batch = self.unflatten_observations(
                self.observations.slice(dim=0, stop=-1).sampler_select(cur_samplers)
            )

            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            norm_adv_targ = []

            for ind in cur_samplers:
                actions_batch.append(self.actions[:, ind])
                prev_actions_batch.append(self.prev_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])

                adv_targ.append(self._advantages[:, ind])
                norm_adv_targ.append(self._normalized_advantages[:, ind])

            actions_batch = torch.stack(actions_batch, 1)  # type:ignore
            prev_actions_batch = torch.stack(prev_actions_batch, 1)  # type:ignore
            value_preds_batch = torch.stack(value_preds_batch, 1)  # type:ignore
            return_batch = torch.stack(return_batch, 1)  # type:ignore
            masks_batch = torch.stack(masks_batch, 1)  # type:ignore
            old_action_log_probs_batch = torch.stack(  # type:ignore
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)  # type:ignore
            norm_adv_targ = torch.stack(norm_adv_targ, 1)  # type:ignore

            yield {
                "observations": observations_batch,
                "memory": memory_batch,
                "actions": su.unflatten(self.action_space, actions_batch),
                "prev_actions": su.unflatten(self.action_space, prev_actions_batch),
                "values": value_preds_batch,
                "returns": return_batch,
                "masks": masks_batch,
                "old_action_log_probs": old_action_log_probs_batch,
                "adv_targ": adv_targ,
                "norm_adv_targ": norm_adv_targ,
                "bsize": int(np.prod(masks_batch.shape[:2])),
            }

    def unflatten_observations(self, flattened_batch: Memory) -> ObservationType:
        result: ObservationType = {}
        for name in flattened_batch:
            full_path = self.flattened_to_unflattened["observations"][name]
            cur_dict = result
            for part in full_path[:-1]:
                if part not in cur_dict:
                    cur_dict[part] = {}
                cur_dict = cast(ObservationType, cur_dict[part])
            cur_dict[full_path[-1]] = flattened_batch[name][0]
        return result

    def pick_observation_step(self, step: int) -> ObservationType:
        return self.unflatten_observations(self.observations.step_select(step))

    def pick_memory_step(self, step: int) -> Memory:
        assert step in [0, self.step, -1], "Can only access the first or last memory."
        return self.memory_first_last.step_squeeze(min(step, 1))

    def pick_prev_actions_step(self, step: int) -> ActionType:
        return su.unflatten(self.action_space, self.prev_actions[step : step + 1])

    def agent_input_for_next_step(self) -> Dict[str, Any]:
        return {
            "observations": self.pick_observation_step(self.step),
            "memory": self.pick_memory_step(self.step),
            "prev_actions": self.pick_prev_actions_step(self.step),
            "masks": self.masks[self.step : self.step + 1],
        }
