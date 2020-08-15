# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import typing
from collections import defaultdict
from typing import Union, List, Dict, Tuple, DefaultDict, Sequence, cast, Any, Optional

import numpy as np
import torch

from core.algorithms.onpolicy_sync.policy import ActorCriticModel
from core.base_abstractions.misc import Memory
from utils.system import get_logger


class RolloutStorage:
    """Class for storing rollout information for RL trainers."""

    FLATTEN_SEPARATOR: str = "._AUTOFLATTEN_."
    DEFAULT_RNN_MEMORY_NAME: str = "._AUTO_RNN_HIDDEN_TO_MEMORY_."  # there can only be one memory with this name
    DEFAULT_RNN_MEMORY_ACCESSOR: str = "auto_rnn_hidden_to_memory"
    DEFAULT_RNN_MEMORY_SAMPLER_AXIS: int = 2  # in actor-critic tensor shape

    def __init__(
        self,
        num_steps: int,
        num_samplers: int,
        actor_critic: ActorCriticModel,
        *args,
        **kwargs,
    ):
        self.num_steps = num_steps

        self.flattened_spaces: Dict[str, Dict[str, List[str]]] = {
            "memory": dict(),
            "observations": dict(),
        }
        self.reverse_flattened_spaces: Dict[str, Dict[Tuple, List[str]]] = {
            "memory": dict(),
            "observations": dict(),
        }

        # self.names = ["step", "sampler", "agent", None]
        self.names = None

        action_space = actor_critic.action_space
        num_recurrent_layers = getattr(actor_critic, "num_recurrent_layers", 0)
        spec = (
            actor_critic.recurrent_hidden_state_size
            if "num_recurrent_layers" in dir(actor_critic)
            else 0
        )  # actually a memory spec if num_rnn_layers is < 0

        self.num_agents = getattr(actor_critic, "num_agents", 1)

        self.memory: Memory = self.create_memory(
            num_recurrent_layers, spec, num_samplers
        )

        self.observations: Memory = Memory()

        self.rewards = torch.zeros(num_steps, num_samplers, self.num_agents, 1,)
        self.value_preds = torch.zeros(num_steps + 1, num_samplers, self.num_agents, 1,)
        self.returns = torch.zeros(num_steps + 1, num_samplers, self.num_agents, 1,)
        self.action_log_probs = torch.zeros(
            num_steps, num_samplers, self.num_agents, 1,
        )
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(
            num_steps, num_samplers, self.num_agents, action_shape,
        )
        self.prev_actions = torch.zeros(
            num_steps + 1, num_samplers, self.num_agents, action_shape,
        )
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.ones(num_steps + 1, num_samplers, self.num_agents, 1,)

        self.step = 0

        self.unnarrow_data: DefaultDict[
            str, Union[int, torch.Tensor, dict]
        ] = defaultdict(dict)

    def create_memory(
        self,
        num_recurrent_layers: int,
        spec: Union[
            Dict[str, Tuple[Sequence[Tuple[str, Optional[int]]], torch.dtype]], int
        ],
        num_samplers: int,
    ):
        if num_recurrent_layers >= 0:
            assert isinstance(spec, int)
            all_dims = [
                self.num_steps + 1,
                num_recurrent_layers,
                num_samplers,
                self.num_agents,
                spec,
            ]
            tensor = torch.zeros(*all_dims, dtype=torch.float32,)
            memory = Memory(
                {
                    self.DEFAULT_RNN_MEMORY_NAME: (
                        tensor,
                        self.DEFAULT_RNN_MEMORY_SAMPLER_AXIS,
                    )
                }
            )
            self.flattened_spaces["memory"][self.DEFAULT_RNN_MEMORY_NAME] = [
                self.DEFAULT_RNN_MEMORY_NAME
            ]
            self.reverse_flattened_spaces["memory"][
                tuple([self.DEFAULT_RNN_MEMORY_ACCESSOR])
            ] = [self.DEFAULT_RNN_MEMORY_NAME]
            return memory
        else:
            assert isinstance(spec, Dict)
            memory = Memory()
            for key in spec:
                dims_template, dtype = spec[key]
                dim_to_pos = {dim[0]: it for it, dim in enumerate(dims_template)}

                assert (
                    "step" in dim_to_pos and dim_to_pos["step"] == 0
                ), "`step` dim must be the first dimension"
                assert (
                    "sampler" in dim_to_pos
                    and dim_to_pos["step"] < dim_to_pos["sampler"]
                ), "`sampler` dim must be after `step` and before `agent`"
                assert (
                    "agent" in dim_to_pos
                    and dim_to_pos["agent"] == dim_to_pos["sampler"] + 1
                ), "`agent` dim must be after `agent`"

                all_dims = [d[1] for d in dims_template]
                all_dims[dim_to_pos["step"]] = self.num_steps + 1
                all_dims[dim_to_pos["sampler"]] = num_samplers

                memory.check_append(
                    key, torch.zeros(*all_dims, dtype=dtype), dim_to_pos["sampler"]
                )
                self.flattened_spaces["memory"][key] = [key]
                self.reverse_flattened_spaces["memory"][(key,)] = [key]  # TODO
            return memory

    def to(self, device: torch.device):
        for sensor in self.observations:
            self.observations[sensor] = (
                self.observations[sensor][0].to(device),
                self.observations[sensor][1],
            )
        for name in self.memory:
            self.memory[name] = (self.memory[name][0].to(device), self.memory[name][1])
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def rnn_to_memory(self, rnn: torch.Tensor):
        return Memory(
            [
                (
                    self.DEFAULT_RNN_MEMORY_NAME,
                    (rnn, self.DEFAULT_RNN_MEMORY_SAMPLER_AXIS),
                )
            ]
        )

    def insert_observations(
        self, observations: Any, time_step: int,
    ):
        self.insert_tensors(
            storage_name="observations", unflattened=observations, time_step=time_step
        )

    def insert_initial_observations(
        self,
        observations: Dict[str, Union[torch.Tensor, Dict, Tuple[torch.Tensor, int]]],
    ):
        self.insert_tensors(
            storage_name="observations", unflattened=observations, time_step=0
        )

    def insert_memory(
        self, memory: Memory, time_step: int,
    ):
        self.insert_tensors(
            storage_name="memory", unflattened=memory, time_step=time_step
        )

    def insert_tensors(
        self,
        storage_name: str,
        unflattened: Any,
        prefix: str = "",
        path: Sequence[str] = (),
        time_step: int = 0,
    ):
        storage = getattr(self, storage_name)
        path = list(path)

        for name in unflattened:
            current_data = unflattened[name]
            if not torch.is_tensor(current_data) and not isinstance(
                current_data, tuple
            ):
                self.insert_tensors(
                    storage_name,
                    current_data,
                    prefix=prefix + name + self.FLATTEN_SEPARATOR,
                    path=path + [name],
                    time_step=time_step,
                )
            else:
                sampler_dim = 1  # dim 0 for step
                if isinstance(current_data, Sequence):
                    sampler_dim = current_data[1]
                    current_data = current_data[0]

                flatten_name = prefix + name
                if flatten_name not in storage:
                    storage[flatten_name] = (
                        torch.zeros_like(current_data)  # type:ignore
                        .repeat(
                            self.num_steps
                            + 1,  # valid for both observations and memory
                            *(1 for _ in range(len(current_data.shape))),
                        )
                        .to(
                            torch.device("cpu")
                            if self.actions.get_device() < 0
                            else self.actions.get_device()
                        ),
                        sampler_dim,
                    )

                    assert (
                        flatten_name not in self.flattened_spaces[storage_name]
                    ), "new flattened name {} already existing in flattened spaces[{}]".format(
                        flatten_name, storage_name
                    )
                    self.flattened_spaces[storage_name][flatten_name] = path + [name]
                    self.reverse_flattened_spaces[storage_name][
                        tuple(path + [name])
                    ] = flatten_name

                storage[flatten_name][0][time_step : time_step + 1].copy_(current_data)

    def insert(
        self,
        observations: Dict[str, Union[torch.Tensor, Dict]],
        memory: Union[torch.Tensor, Memory],
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        *args,
    ):
        assert len(args) == 0

        self.insert_observations(observations, time_step=self.step + 1)

        if memory is not None:
            # Support for single RNN memory
            if isinstance(memory, torch.Tensor):
                memory = self.rnn_to_memory(memory)

            self.insert_memory(cast(Memory, memory), time_step=self.step + 1)
        else:
            assert self.memory[self.DEFAULT_RNN_MEMORY_NAME][0].shape[-1] == 0  # 0 dims

        self.actions[self.step : self.step + 1].copy_(actions)  # type:ignore
        self.prev_actions[self.step + 1 : self.step + 2].copy_(actions)  # type:ignore
        self.action_log_probs[self.step : self.step + 1].copy_(  # type:ignore
            action_log_probs
        )
        self.value_preds[self.step : self.step + 1].copy_(value_preds)  # type:ignore
        self.rewards[self.step : self.step + 1].copy_(rewards)  # type:ignore
        self.masks[self.step + 1 : self.step + 2].copy_(masks)  # type:ignore

        self.step = (self.step + 1) % self.num_steps

    def reshape(self, keep_list: Sequence[int]):
        keep_list = list(keep_list)
        if self.actions.shape[1] == len(keep_list):  # samplers dim
            return  # we are keeping everything, no need to copy

        self.observations = self.observations.index_select(keep_list)
        self.memory = self.memory.index_select(keep_list)
        self.actions = self.actions[:, keep_list]
        self.prev_actions = self.prev_actions[:, keep_list]
        self.action_log_probs = self.action_log_probs[:, keep_list]
        self.value_preds = self.value_preds[:, keep_list]
        self.rewards = self.rewards[:, keep_list]
        self.masks = self.masks[:, keep_list]
        self.returns = self.returns[:, keep_list]

    def narrow(self):
        assert len(self.unnarrow_data) == 0, "attempting to narrow narrowed rollouts"

        if self.step == 0:  # we're actually done
            get_logger().warning("Called narrow with self.step == 0")
            return

        for sensor in self.observations:
            self.unnarrow_data["observations"][sensor] = self.observations[sensor][0]
            self.observations[sensor] = (
                self.observations[sensor][0].narrow(0, 0, self.step + 1),
                self.observations[sensor][1],
            )

        for name in self.memory:
            self.unnarrow_data["memory"][name] = self.memory[name][0]
            self.memory[name] = (
                self.memory[name][0].narrow(0, 0, self.step + 1),
                self.memory[name][1],
            )

        for name in ["prev_actions", "value_preds", "returns", "masks"]:
            self.unnarrow_data[name] = getattr(self, name)
            setattr(self, name, self.unnarrow_data[name].narrow(0, 0, self.step + 1))

        for name in ["actions", "action_log_probs", "rewards"]:
            self.unnarrow_data[name] = getattr(self, name)
            setattr(self, name, self.unnarrow_data[name].narrow(0, 0, self.step))

        self.unnarrow_data["num_steps"] = self.num_steps
        self.num_steps = self.step
        self.step = 0

    def unnarrow(self):
        assert len(self.unnarrow_data) > 0, "attempting to unnarrow unnarrowed rollouts"

        for sensor in self.observations:
            self.observations[sensor] = (
                self.unnarrow_data["observations"][sensor],
                self.observations[sensor][1],
            )
            del self.unnarrow_data["observations"][sensor]

        assert (
            len(self.unnarrow_data["observations"]) == 0
        ), "unnarrow_data contains observations {}".format(
            self.unnarrow_data["observations"]
        )
        del self.unnarrow_data["observations"]

        for name in self.memory:
            self.memory[name] = (
                self.unnarrow_data["memory"][name],
                self.memory[name][1],
            )
            del self.unnarrow_data["memory"][name]

        assert (
            len(self.unnarrow_data["memory"]) == 0
        ), "unnarrow_data contains memory {}".format(self.unnarrow_data["memory"])
        del self.unnarrow_data["memory"]

        for name in [
            "prev_actions",
            "value_preds",
            "returns",
            "masks",
            "actions",
            "action_log_probs",
            "rewards",
        ]:
            setattr(self, name, self.unnarrow_data[name])
            del self.unnarrow_data[name]

        self.num_steps = self.unnarrow_data["num_steps"]
        del self.unnarrow_data["num_steps"]

        assert len(self.unnarrow_data) == 0

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0][0].copy_(self.observations[sensor][0][-1])

        for name in self.memory:
            self.memory[name][0][0].copy_(self.memory[name][0][-1])

        self.masks[0].copy_(self.masks[-1])
        self.prev_actions[0].copy_(self.prev_actions[-1])

        if len(self.unnarrow_data) > 0:
            self.unnarrow()

    def compute_returns(
        self, next_value: torch.Tensor, use_gae: bool, gamma: float, tau: float
    ):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae  # type:ignore
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages: torch.Tensor, num_mini_batch: int):
        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5
        )

        num_samplers = self.rewards.size(1)
        assert num_samplers >= num_mini_batch, (
            "The number of processes ({}) "
            "must be greater than or equal to the number of "
            "mini batches ({}).".format(num_samplers, num_mini_batch)
        )

        inds = np.round(
            np.linspace(0, num_samplers, num_mini_batch + 1, endpoint=True)
        ).astype(np.int32)
        pairs = list(zip(inds[:-1], inds[1:]))
        random.shuffle(pairs)

        for start_ind, end_ind in pairs:
            observations_batch = defaultdict(list)
            memory_batch = defaultdict(list)

            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            norm_adv_targ = []

            for ind in range(start_ind, end_ind):
                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][0][:-1, ind]
                    )

                # recurrent_hidden_states_batch.append(
                #     self.recurrent_hidden_states[0, :, ind]
                # )
                for name in self.memory:
                    memory_batch[name].append(
                        self.memory[name][0]
                        .index_select(
                            dim=self.memory[name][1],
                            index=torch.as_tensor(
                                [ind],
                                dtype=torch.int64,
                                device=self.memory[name][0].device,
                            ),
                        )
                        .squeeze(self.memory[name][1])[
                            :1, ...
                        ],  # keep only initial memory!
                    )

                actions_batch.append(self.actions[:, ind])
                prev_actions_batch.append(self.prev_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])

                adv_targ.append(advantages[:, ind])
                norm_adv_targ.append(normalized_advantages[:, ind])

            # T, N = self.num_steps, end_ind - start_ind

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                # noinspection PyTypeChecker
                observations_batch[sensor] = torch.stack(  # type:ignore
                    observations_batch[sensor], 1
                )  # new sampler dimension

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

            # # States is just a (num_recurrent_layers, N, -1) tensor
            # recurrent_hidden_states_batch = torch.stack(
            #     recurrent_hidden_states_batch, 1
            # )
            for name in memory_batch:
                # noinspection PyTypeChecker
                memory_batch[name] = torch.stack(  # type:ignore
                    memory_batch[name],
                    self.memory[name][
                        1
                    ],  # - 1, we don't squeeze out the single step dim, so no need to subtract 1
                )  # actor-critic sampler axis

            # # Flatten the (T, N, ...) tensors to (T * N, ...)
            # for sensor in observations_batch:
            #     # noinspection PyTypeChecker
            #     observations_batch[sensor] = self._flatten_helper(  # type:ignore
            #         t=T, n=N, tensor=cast(torch.Tensor, observations_batch[sensor]),
            #     )
            #
            # actions_batch = self._flatten_helper(T, N, actions_batch)  # type:ignore
            # prev_actions_batch = self._flatten_helper(  # type:ignore
            #     T, N, cast(torch.Tensor, prev_actions_batch)
            # )
            # value_preds_batch = self._flatten_helper(  # type:ignore
            #     T, N, cast(torch.Tensor, value_preds_batch)
            # )
            # return_batch = self._flatten_helper(T, N, return_batch)  # type:ignore
            # masks_batch = self._flatten_helper(T, N, masks_batch)  # type:ignore
            # old_action_log_probs_batch = self._flatten_helper(  # type:ignore
            #     T, N, cast(torch.Tensor, old_action_log_probs_batch)
            # )
            # adv_targ = self._flatten_helper(T, N, adv_targ)  # type:ignore
            # norm_adv_targ = self._flatten_helper(T, N, norm_adv_targ)  # type:ignore

            yield {
                "observations": self.unflatten_batch(
                    observations_batch, "observations"
                ),
                "memory": self.unflatten_batch(memory_batch, "memory"),
                "actions": actions_batch,
                "prev_actions": prev_actions_batch,
                "values": value_preds_batch,
                "returns": return_batch,
                "masks": masks_batch,
                "old_action_log_probs": old_action_log_probs_batch,
                "adv_targ": adv_targ,
                "norm_adv_targ": norm_adv_targ,
            }

    def unflatten_batch(self, flattened_batch: Dict, storage_type: str):
        def ddict2dict(d: Dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = ddict2dict(v)
            return dict(d)

        if storage_type == "memory":
            if len(flattened_batch) == 1:
                key = list(flattened_batch.keys())[0]
                if key == self.DEFAULT_RNN_MEMORY_NAME:
                    return flattened_batch[key]

        result: Dict = defaultdict()
        for name in flattened_batch:
            full_path = self.flattened_spaces[storage_type][name]
            cur_dict = result
            for part in full_path[:-1]:
                cur_dict = cur_dict[part]
            if storage_type == "observations":
                cur_dict[full_path[-1]] = flattened_batch[name]
            else:  # memory
                cur_dict[full_path[-1]] = (
                    flattened_batch[name],
                    self.memory[name][1] - 1,
                )
        return ddict2dict(result) if storage_type == "observations" else Memory(result)

    def pick_step(
        self, step: int, storage_type: str
    ) -> Dict[str, Union[Dict, torch.Tensor]]:
        storage = getattr(self, storage_type)
        if step >= 0:
            batch = {key: storage[key][0][step : step + 1] for key in storage}
        else:
            batch = {key: storage[key][0][step:] for key in storage}
        return self.unflatten_batch(batch, storage_type)

    def pick_observation_step(self, step: int) -> Dict[str, Union[Dict, torch.Tensor]]:
        # observations_batch = {sensor: self.observations[sensor][step] for sensor in self.observations}
        # return self.unflatten_spaces(observations_batch)
        return self.pick_step(step, "observations")

    def pick_memory_step(self, step: int) -> Dict[str, Union[Dict, torch.Tensor]]:
        # memory_batch = {name: self.memory[name][step] for name in self.observations}
        # return self.unflatten_spaces(observations_batch)
        return self.pick_step(step, "memory")

    # @staticmethod
    # def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
    #     """Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).
    #
    #     Args:
    #         t: first dimension of tensor.
    #         n: second dimension of tensor.
    #         tensor: target tensor to be flattened.
    #
    #     Returns:
    #         flattened tensor of size (t*n, ...)
    #     """
    #     return tensor.view(t * n, *tensor.size()[2:])
