# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Union, Dict, Tuple, Sequence, Optional

import numpy as np
import torch
from gym import spaces

from core.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    FullMemorySpecType,
    ObservationType,
)
from core.base_abstractions.misc import Memory
from utils.system import get_logger

ActionType = ObservationType

NUMPY_TO_TORCH_DTYPE_DICT = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


# TODO support hierarchical memory spec
class RolloutStorage(object):
    """Class for storing rollout information for RL trainers."""

    def __init__(
        self, num_steps: int, num_samplers: int, actor_critic: ActorCriticModel,
    ):
        self.num_steps = num_steps
        self.num_samplers = num_samplers
        self.num_agents = getattr(actor_critic, "num_agents", 1)

        self.dim_names = ["step", "sampler", "agent"]
        self.sampler_dim = self.dim_names.index("sampler")

        self.storage = Memory()
        self.unnarrow_storage: Optional[Memory] = None

        self.storage["observations"] = Memory()
        self.storage["memory"] = self.create_memory(
            actor_critic.recurrent_memory_specification
        )
        self.storage["actions"], self.storage["prev_actions"] = self.create_actions(
            actor_critic.action_space
        )

        self.storage.check_append(
            "rewards",
            torch.zeros(num_steps, num_samplers, self.num_agents, 1,),
            self.sampler_dim,
        )
        self.storage.check_append(
            "value_preds",
            torch.zeros(num_steps + 1, num_samplers, self.num_agents, 1,),
            self.sampler_dim,
        )
        self.storage.check_append(
            "returns",
            torch.zeros(num_steps + 1, num_samplers, self.num_agents, 1,),
            self.sampler_dim,
        )
        self.storage.check_append(
            "action_log_probs",
            torch.zeros(num_steps, num_samplers, self.num_agents, 1),
            self.sampler_dim,
        )
        self.storage.check_append(
            "masks",
            torch.ones(num_steps + 1, num_samplers, self.num_agents, 1,),
            self.sampler_dim,
        )

        self.step = 0

    def create_actions(self, action_space: spaces.Space) -> Tuple[Memory, Memory]:
        actions, prev_actions = Memory(), Memory()
        num_steps = self.num_steps
        num_samplers = self.num_samplers

        def dfs(current_action_space, key=[]):
            if isinstance(current_action_space, spaces.Tuple):
                for subkey, subspace in enumerate(current_action_space):
                    dfs(subspace, key + [subkey])
            elif isinstance(current_action_space, spaces.Dict):
                for subkey in current_action_space:
                    dfs(current_action_space[subkey], key + [subkey])
            else:
                if isinstance(current_action_space, spaces.Discrete):
                    action_shape = (1,)
                else:
                    action_shape = current_action_space.shape

                actions.check_append(
                    key=key,
                    tensor=torch.zeros(
                        num_steps,
                        num_samplers,
                        *action_shape,
                        dtype=NUMPY_TO_TORCH_DTYPE_DICT[current_action_space.dtype],
                    ),
                    sampler_dim=self.sampler_dim,
                )
                prev_actions.check_append(
                    key=key,
                    tensor=torch.zeros(
                        num_steps + 1,
                        num_samplers,
                        *action_shape,
                        dtype=NUMPY_TO_TORCH_DTYPE_DICT[current_action_space.dtype],
                    ),
                    sampler_dim=self.sampler_dim,
                )

        dfs(action_space)

        return actions, prev_actions

    # TODO update memory spec and enable recursive memory
    def create_memory(self, spec: Optional[FullMemorySpecType]) -> Memory:
        if spec is None:
            return Memory()

        memory = Memory()
        for key in spec:
            dims_template, dtype = spec[key]

            dim_names = ["step"] + [d[0] for d in dims_template]
            sampler_dim = dim_names.index("sampler")

            all_dims = [self.num_steps + 1] + [d[1] for d in dims_template]
            all_dims[sampler_dim] = self.num_samplers

            memory.check_append(
                key=key,
                tensor=torch.zeros(*all_dims, dtype=dtype),
                sampler_dim=sampler_dim,
            )

        return memory

    def to(self, device: torch.device):
        self.storage.to(device)

    def _store(
        self,
        storage_name: str,
        unflattened: Union[ObservationType, Memory, ActionType, torch.Tensor],
        time_step: int = 0,
    ):
        storage = self.storage[storage_name]
        default_sampler_dim = self.sampler_dim
        current_device: Optional[torch.device] = None
        if storage_name == "observations":
            current_device = self.storage.tensor("rewards").get_device()

        def dfs(current_data, path=[]):
            if isinstance(current_data, Tuple) and storage_name != "memory":
                for subkey, subdata in enumerate(current_data):
                    dfs(subdata, path + [subkey])
            elif isinstance(current_data, Dict):
                for subkey in current_data:
                    dfs(current_data[subkey], path + [subkey])
            else:
                sampler_dim = default_sampler_dim
                if isinstance(current_data, Tuple):
                    assert storage_name == "memory"
                    sampler_dim = current_data[1]
                    current_data = current_data[0]

                if path not in storage:
                    assert storage_name == "observations"
                    storage.append_check(
                        path,
                        torch.zeros_like(current_data)  # type:ignore
                        .repeat(
                            self.num_steps
                            + 1,  # required for observations (and others)
                            *(1 for _ in range(len(current_data.shape))),
                        )
                        .to(
                            torch.device("cpu")
                            if current_device < 0
                            else current_device
                        ),
                        sampler_dim,
                    )

                if storage_name == "memory":
                    # current_data does not have a step dimension (memory)
                    storage.tensor(path)[time_step].copy_(current_data)
                else:
                    # current_data has a step dimension (all except for memory)
                    assert time_step >= 0
                    storage.tensor(path)[time_step : time_step + 1].copy_(current_data)

        dfs(unflattened)

    def insert(
        self,
        observations: ObservationType,
        memory: Optional[Memory],
        actions: ActionType,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ):
        if memory is None:
            assert len(self.storage["memory"]) == 0
            return
        self._store("memory", memory, self.step + 1)

        self._store("observations", observations, self.step + 1)
        self._store("actions", actions, self.step)
        self._store("prev_actions", actions, self.step + 1)

        self._store("action_log_probs", action_log_probs, self.step)
        self._store("value_preds", value_preds, self.step)
        self._store("rewards", rewards, self.step)
        self._store("masks", masks, self.step + 1)

        self.step = (self.step + 1) % self.num_steps

    def sampler_select(self, keep_list: Sequence[int]):
        keep_list = list(keep_list)
        if self.storage.tensor("rewards").shape[self.sampler_dim] == len(keep_list):
            return  # we are keeping everything, no need to copy

        self.storage.sampler_select(keep_list)

    def narrow(self):
        assert self.unnarrow_storage is None, "attempting to narrow narrowed rollouts"

        if self.step == 0:  # we're actually done
            get_logger().warning("Called narrow with self.step == 0")
            return

        self.unnarrow_storage = Memory()

        for name in [
            "prev_actions",
            "value_preds",
            "returns",
            "masks",
            "observations",
            "memory",
        ]:
            self.unnarrow_storage[name] = self.storage.narrow_steps(
                self.step + 1, key=name
            )

        for name in ["actions", "action_log_probs", "rewards"]:
            self.unnarrow_storage[name] = self.storage.narrow_steps(self.step, key=name)

        assert len(self.storage) == len(
            self.unnarrow_storage
        ), "Different number of tensors after narrow: storage {} unnarrow_storage {}".format(
            len(self.storage), len(self.unnarrow_storage)
        )

        self.unnarrow_storage["num_steps"] = self.num_steps
        self.num_steps = self.step

        # We just finished a rollout, so we reset the step counter for the next one:
        self.step = 0

    def unnarrow(self):
        assert (
            self.unnarrow_storage is not None
        ), "Attempting to unnarrow unnarrowed rollouts"

        self.storage = self.unnarrow_storage
        self.num_steps = self.storage.pop("num_steps")
        self.unnarrow_storage = None

    def after_update(self):
        def cycler(node, key, path, *args, **kwargs):
            node.tensor(key)[0].copy_(node.tensor(key)[-1])

        for name in ["observations", "memory", "masks", "prev_actions"]:
            mem, key = self.storage.traverse_to_parent(name, create_children=False)
            self.storage.traversal(mem, cycler, topkey=key)

        if self.unnarrow_storage is not None:
            self.unnarrow()

    def compute_returns(
        self, next_value: torch.Tensor, use_gae: bool, gamma: float, tau: float
    ):
        rewards = self.storage.tensor("rewards")
        masks = self.storage.tensor("masks")
        returns = self.storage.tensor("returns")

        if use_gae:
            value_preds = self.storage.tensor("value_preds")
            value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(rewards.size(0))):
                delta = (
                    rewards[step]
                    + gamma * value_preds[step + 1] * masks[step + 1]
                    - value_preds[step]
                )
                gae = delta + gamma * tau * masks[step + 1] * gae  # type:ignore
                returns[step] = gae + value_preds[step]
        else:
            returns[-1] = next_value
            for step in reversed(range(rewards.size(0))):
                returns[step] = (
                    returns[step + 1] * gamma * masks[step + 1] + rewards[step]
                )

    def recurrent_generator(self, advantages: torch.Tensor, num_mini_batch: int):
        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5
        )

        num_samplers = self.storage.tensor("rewards").size(1)
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

            memory_batch = self.storage.step_squeeze(0, key="memory").sampler_select(
                cur_samplers
            )
            observations_batch = self.storage.slice(
                dim=0, stop=-1, key="observations"
            ).sampler_select(cur_samplers)

            # TODO convert actions/prev_actions into usable format (is Memory good enough?)
            actions_batch = self.storage.slice(dim=0, key="actions").sampler_select(
                cur_samplers
            )
            prev_actions_batch = self.storage.slice(
                dim=0, stop=-1, key="prev_actions"
            ).sampler_select(cur_samplers)

            value_preds_batch = (
                self.storage.slice(dim=0, stop=-1, key="value_preds")
                .sampler_select(cur_samplers)
                .tensor("value_preds")
            )

            returns_batch = (
                self.storage.slice(dim=0, stop=-1, key="returns")
                .sampler_select(cur_samplers)
                .tensor("returns")
            )

            masks_batch = (
                self.storage.slice(dim=0, stop=-1, key="masks")
                .sampler_select(cur_samplers)
                .tensor("masks")
            )

            old_action_log_probs_batch = (
                self.storage.slice(dim=0, key="action_log_probs")
                .sampler_select(cur_samplers)
                .tensor("action_log_probs")
            )

            adv_targ = []
            norm_adv_targ = []
            for ind in cur_samplers:
                adv_targ.append(advantages[:, ind])
                norm_adv_targ.append(normalized_advantages[:, ind])

            adv_targ = torch.stack(adv_targ, 1)  # type:ignore
            norm_adv_targ = torch.stack(norm_adv_targ, 1)  # type:ignore

            yield {
                "observations": observations_batch,
                "memory": memory_batch,
                "actions": actions_batch,
                "prev_actions": prev_actions_batch,
                "values": value_preds_batch,
                "returns": returns_batch,
                "masks": masks_batch,
                "old_action_log_probs": old_action_log_probs_batch,
                "adv_targ": adv_targ,
                "norm_adv_targ": norm_adv_targ,
            }

    def pick_observation_step(self, step: int) -> ObservationType:
        return self.storage["observations"].step_select(step)

    def pick_memory_step(self, step: int) -> Memory:
        return self.storage["memory"].step_squeeze(step)
