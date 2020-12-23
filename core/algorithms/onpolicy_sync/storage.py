# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from collections import defaultdict
from typing import Union, List, Dict, Tuple, DefaultDict, Sequence, cast, Optional

import numpy as np
import torch

from core.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    FullMemorySpecType,
    ObservationType,
)
from core.base_abstractions.misc import Memory
from utils.system import get_logger
from utils.spaces_utils import flatdim


class RolloutStorage(object):
    """Class for storing rollout information for RL trainers."""

    FLATTEN_SEPARATOR: str = "._AUTOFLATTEN_."

    def __init__(
        self, num_steps: int, num_samplers: int, actor_critic: ActorCriticModel,
    ):
        self.num_steps = num_steps

        self.flattened_to_unflattened: Dict[str, Dict[str, List[str]]] = {
            "memory": dict(),
            "observations": dict(),
        }
        self.unflattened_to_flattened: Dict[str, Dict[Tuple[str, ...], str]] = {
            "memory": dict(),
            "observations": dict(),
        }

        self.dim_names = ["step", "sampler", None]

        self.memory: Memory = self.create_memory(
            actor_critic.recurrent_memory_specification, num_samplers
        )
        self.observations: Memory = Memory()

        self.num_agents = getattr(actor_critic, "num_agents", 1)

        self.rewards: Optional[torch.Tensor] = None
        self.value_preds: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None
        self.action_log_probs: Optional[torch.Tensor] = None

        self.masks = torch.ones(num_steps + 1, num_samplers, 1)

        action_shape = flatdim(actor_critic.action_space)
        self.actions = torch.zeros(num_steps, num_samplers, action_shape,)
        self.prev_actions = torch.zeros(num_steps + 1, num_samplers, action_shape,)

        self.step = 0

        self.unnarrow_data: DefaultDict[
            str, Union[int, torch.Tensor, Dict]
        ] = defaultdict(dict)

        self.device = "cpu"

    def create_memory(
        self, spec: Optional[FullMemorySpecType], num_samplers: int,
    ) -> Memory:
        if spec is None:
            return Memory()

        memory = Memory()
        for key in spec:
            dims_template, dtype = spec[key]

            dim_names = ["step"] + [d[0] for d in dims_template]
            sampler_dim = dim_names.index("sampler")

            all_dims = [self.num_steps + 1] + [d[1] for d in dims_template]
            all_dims[sampler_dim] = num_samplers

            memory.check_append(
                key=key,
                tensor=torch.zeros(*all_dims, dtype=dtype),
                sampler_dim=sampler_dim,
            )

            self.flattened_to_unflattened["memory"][key] = [key]
            self.unflattened_to_flattened["memory"][(key,)] = key

        return memory

    def to(self, device: torch.device):
        self.observations.to(device)
        self.memory.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

        if self.rewards is not None:
            self.rewards = self.rewards.to(device)
            self.value_preds = self.value_preds.to(device)
            self.action_log_probs = self.action_log_probs.to(device)

        if self.returns is not None:
            self.returns = self.returns.to(device)

        self.device = device

    def insert_observations(
        self, observations: ObservationType, time_step: int = 0,
    ):
        self.insert_tensors(
            storage_name="observations", unflattened=observations, time_step=time_step
        )

    def insert_memory(
        self, memory: Optional[Memory], time_step: int,
    ):
        if memory is None:
            assert len(self.memory) == 0
            return
        self.insert_tensors(
            storage_name="memory", unflattened=memory, time_step=time_step
        )

    def insert_tensors(
        self,
        storage_name: str,
        unflattened: Union[ObservationType, Memory],
        prefix: str = "",
        path: Sequence[str] = (),
        time_step: int = 0,
    ):
        storage = getattr(self, storage_name)
        path = list(path)

        for name in unflattened:
            current_data = unflattened[name]

            if isinstance(current_data, Dict):
                self.insert_tensors(
                    storage_name,
                    cast(ObservationType, current_data),
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
                        self.num_steps + 1,  # required for observations (and memory)
                        *(1 for _ in range(len(current_data.shape))),
                    )
                    .to(self.device),
                    sampler_dim,
                )

                assert (
                    flatten_name not in self.flattened_to_unflattened[storage_name]
                ), "new flattened name {} already existing in flattened spaces[{}]".format(
                    flatten_name, storage_name
                )
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
            else:
                # current_data does not have a step dimension
                storage[flatten_name][0][time_step].copy_(current_data)

    def create_tensor_storage(self, num_steps: int, step: torch.Tensor) -> torch.Tensor:
        # print("CREATE", step.shape)
        return (
            torch.zeros_like(step)
            .repeat(num_steps, *(1 for _ in range(len(step.shape[1:]))),)
            .to(self.device)
        )

    def insert(
        self,
        observations: ObservationType,
        memory: Optional[Memory],
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ):
        self.insert_observations(observations, time_step=self.step + 1)
        self.insert_memory(memory, time_step=self.step + 1)

        self.actions[self.step : self.step + 1].copy_(actions)  # type:ignore
        self.prev_actions[self.step + 1 : self.step + 2].copy_(actions)  # type:ignore

        self.masks[self.step + 1 : self.step + 2].copy_(masks)  # type:ignore

        if self.rewards is None:
            self.rewards = self.create_tensor_storage(self.num_steps, rewards)
            self.value_preds = self.create_tensor_storage(
                self.num_steps + 1, value_preds
            )
            self.action_log_probs = self.create_tensor_storage(
                self.num_steps, action_log_probs
            )
            self.returns = self.create_tensor_storage(self.num_steps + 1, value_preds)

        self.action_log_probs[self.step : self.step + 1].copy_(  # type:ignore
            action_log_probs
        )
        self.value_preds[self.step : self.step + 1].copy_(value_preds)  # type:ignore
        self.rewards[self.step : self.step + 1].copy_(rewards)  # type:ignore

        self.step = (self.step + 1) % self.num_steps

    def sampler_select(self, keep_list: Sequence[int]):
        keep_list = list(keep_list)
        if self.actions.shape[1] == len(keep_list):  # samplers dim
            return  # we are keeping everything, no need to copy

        self.observations = self.observations.sampler_select(keep_list)
        self.memory = self.memory.sampler_select(keep_list)
        self.actions = self.actions[:, keep_list]
        self.prev_actions = self.prev_actions[:, keep_list]
        self.masks = self.masks[:, keep_list]

        if self.rewards is not None:
            self.action_log_probs = self.action_log_probs[:, keep_list]
            self.value_preds = self.value_preds[:, keep_list]
            self.rewards = self.rewards[:, keep_list]
            self.returns = self.returns[:, keep_list]

    def narrow(self):
        assert len(self.unnarrow_data) == 0, "attempting to narrow narrowed rollouts"

        if self.step == 0:  # we're actually done
            get_logger().warning("Called narrow with self.step == 0")
            return

        for storage_name in ["observations", "memory"]:
            storage: Memory = getattr(self, storage_name)
            for key in storage:
                self.unnarrow_data[storage_name][key] = storage.tensor(key)
                storage[key] = (
                    storage.tensor(key).narrow(
                        dimension=0, start=0, length=self.step + 1
                    ),
                    storage.sampler_dim(key),
                )

        for name in ["prev_actions", "value_preds", "returns", "masks"]:
            if getattr(self, name) is not None:
                self.unnarrow_data[name] = getattr(self, name)
                setattr(
                    self,
                    name,
                    self.unnarrow_data[name].narrow(
                        dimension=0, start=0, length=self.step + 1
                    ),
                )

        for name in ["actions", "action_log_probs", "rewards"]:
            if getattr(self, name) is not None:
                self.unnarrow_data[name] = getattr(self, name)
                setattr(
                    self,
                    name,
                    self.unnarrow_data[name].narrow(
                        dimension=0, start=0, length=self.step
                    ),
                )

        self.unnarrow_data["num_steps"] = self.num_steps
        self.num_steps = self.step
        self.step = 0  # we just finished a rollout, so we reset it for the next one

    def unnarrow(self):
        assert len(self.unnarrow_data) > 0, "attempting to unnarrow unnarrowed rollouts"

        for storage_name in ["observations", "memory"]:
            storage: Memory = getattr(self, storage_name)
            for key in storage:
                storage[key] = (
                    self.unnarrow_data[storage_name][key],
                    storage.sampler_dim(key),
                )
                self.unnarrow_data[storage_name].pop(key)

            # Note that memory can be empty
            assert (
                storage_name not in self.unnarrow_data
                or len(self.unnarrow_data[storage_name]) == 0
            ), "unnarrow_data contains {} {}".format(
                storage_name, self.unnarrow_data[storage_name]
            )
            self.unnarrow_data.pop(storage_name, None)

        for name in [
            "prev_actions",
            "value_preds",
            "returns",
            "masks",
            "actions",
            "action_log_probs",
            "rewards",
        ]:
            if name in self.unnarrow_data:
                setattr(self, name, self.unnarrow_data[name])
                self.unnarrow_data.pop(name)

        self.num_steps = self.unnarrow_data["num_steps"]
        self.unnarrow_data.pop("num_steps")

        assert len(self.unnarrow_data) == 0

    def after_update(self):
        for storage in [self.observations, self.memory]:
            for key in storage:
                storage[key][0][0].copy_(storage[key][0][-1])

        self.masks[0].copy_(self.masks[-1])
        self.prev_actions[0].copy_(self.prev_actions[-1])

        if len(self.unnarrow_data) > 0:
            self.unnarrow()

    def compute_returns(
        self, next_value: torch.Tensor, use_gae: bool, gamma: float, tau: float
    ):
        mask_step_shape = self.masks.shape[1:] + (1,) * (
            len(self.value_preds.shape) - len(self.masks.shape)
        )

        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.shape[0])):
                delta = (
                    self.rewards[step]
                    + gamma
                    * self.value_preds[step + 1]
                    * self.masks[step + 1].view(mask_step_shape)
                    - self.value_preds[step]
                )
                gae = (
                    delta
                    + gamma * tau * self.masks[step + 1].view(mask_step_shape) * gae
                )  # type:ignore
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.shape[0])):
                self.returns[step] = (
                    self.returns[step + 1]
                    * gamma
                    * self.masks[step + 1].view(mask_step_shape)
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages: torch.Tensor, num_mini_batch: int):
        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5
        )

        num_samplers = self.rewards.shape[1]
        assert num_samplers >= num_mini_batch, (
            "The number of task samplers ({}) "
            "must be greater than or equal to the number of "
            "mini batches ({}).".format(num_samplers, num_mini_batch)
        )

        inds = np.round(
            np.linspace(0, num_samplers, num_mini_batch + 1, endpoint=True)
        ).astype(np.int32)
        pairs = list(zip(inds[:-1], inds[1:]))
        random.shuffle(pairs)

        for start_ind, end_ind in pairs:
            cur_samplers = list(range(start_ind, end_ind))

            memory_batch = self.memory.step_squeeze(0).sampler_select(cur_samplers)
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

                adv_targ.append(advantages[:, ind])
                norm_adv_targ.append(normalized_advantages[:, ind])

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
                "actions": actions_batch,
                "prev_actions": prev_actions_batch,
                "values": value_preds_batch,
                "returns": return_batch,
                "masks": masks_batch,
                "old_action_log_probs": old_action_log_probs_batch,
                "adv_targ": adv_targ,
                "norm_adv_targ": norm_adv_targ,
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
        return self.memory.step_squeeze(step)
