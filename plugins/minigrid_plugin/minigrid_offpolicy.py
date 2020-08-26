import os
import queue
from collections import defaultdict
import random
from typing import Dict, Tuple, Any, cast, Iterator, List, Union, Optional

import numpy as np
import torch
from gym_minigrid.minigrid import MiniGridEnv
import babyai

from core.algorithms.offpolicy_sync.losses.abstract_offpolicy_loss import (
    AbstractOffPolicyLoss,
    Memory,
)
from core.algorithms.onpolicy_sync.policy import ActorCriticModel, ObservationType
from plugins.minigrid_plugin.minigrid_sensors import MiniGridMissionSensor
from utils.system import get_logger

_DATASET_CACHE: Dict[str, Any] = {}


class MiniGridOffPolicyExpertCELoss(AbstractOffPolicyLoss[ActorCriticModel]):
    def __init__(self, total_episodes_in_epoch: Optional[int] = None):
        super().__init__()
        self.total_episodes_in_epoch = total_episodes_in_epoch

    def loss(  # type:ignore
        self,
        model: ActorCriticModel,
        batch: ObservationType,
        memory: Memory,
        *args,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Dict[str, float], Memory, int]:
        rollout_len, nrollouts = cast(torch.Tensor, batch["minigrid_ego_image"]).shape[
            :2
        ]

        if len(memory) == 0:
            spec = model.recurrent_memory_specification
            for key in spec:
                dims_template, dtype = spec[key]

                dim_names = [d[0] for d in dims_template]
                sampler_dim = dim_names.index("sampler")

                all_dims = [d[1] for d in dims_template]
                all_dims[sampler_dim] = nrollouts

                memory.check_append(
                    key=key,
                    tensor=torch.zeros(
                        *all_dims,
                        dtype=dtype,
                        device=cast(torch.Tensor, batch["minigrid_ego_image"]).device
                    ),
                    sampler_dim=sampler_dim,
                )

        ac_out, memory = model.forward(
            observations=batch,
            memory=memory,
            prev_actions=None,  # type:ignore
            masks=cast(torch.FloatTensor, batch["masks"]),
        )

        expert_ce_loss = -ac_out.distributions.log_probs(batch["expert_action"]).mean()

        info = {"expert_ce": expert_ce_loss.item()}

        if self.total_episodes_in_epoch is not None:
            if "completed_episode_count" not in memory:
                memory["completed_episode_count"] = 0
            memory["completed_episode_count"] += (
                int(np.prod(batch["masks"].shape))  # type:ignore
                - batch["masks"].sum().item()  # type:ignore
            )
            info["epoch_progress"] = (
                memory["completed_episode_count"] / self.total_episodes_in_epoch
            )

        return expert_ce_loss, info, memory, rollout_len * nrollouts


class ExpertTrajectoryIterator(Iterator):
    def __init__(
        self,
        data: List[Tuple[str, bytes, List[int], MiniGridEnv.Actions]],
        nrollouts: int,
        rollout_len: int,
        instr_len: Optional[int],
        restrict_max_steps_in_dataset: Optional[int] = None,
    ):
        super(ExpertTrajectoryIterator, self).__init__()
        self.restrict_max_steps_in_dataset = restrict_max_steps_in_dataset

        if restrict_max_steps_in_dataset is not None:
            restricted_data = []
            cur_len = 0
            for i, d in enumerate(data):
                if cur_len >= restrict_max_steps_in_dataset:
                    break
                restricted_data.append(d)
                cur_len += len(d[2])
            data = restricted_data
            if cur_len > restrict_max_steps_in_dataset:
                # throw away the last steps in the last trajec
                data[-1] = data[-1][: restrict_max_steps_in_dataset - cur_len]

        self.data = data
        self.trajectory_inds = list(range(len(data)))
        self.instr_len = instr_len
        random.shuffle(self.trajectory_inds)

        assert nrollouts <= len(self.trajectory_inds), "Too many rollouts requested."

        self.nrollouts = nrollouts
        self.rollout_len = rollout_len

        self.rollout_queues: List[queue.Queue] = [
            queue.Queue() for _ in range(nrollouts)
        ]
        for q in self.rollout_queues:
            self.add_data_to_rollout_queue(q)

        self.minigrid_mission_sensor: Optional[MiniGridMissionSensor] = None
        if instr_len is not None:
            self.minigrid_mission_sensor = MiniGridMissionSensor(instr_len)

    def add_data_to_rollout_queue(self, q: queue.Queue) -> bool:
        assert q.empty()
        if len(self.trajectory_inds) == 0:
            return False

        for i, step in enumerate(
            babyai.utils.demos.transform_demos([self.data[self.trajectory_inds.pop()]])[
                0
            ]
        ):
            q.put((*step, i == 0))

        return True

    def get_data_for_rollout_ind(self, rollout_ind: int) -> Dict[str, np.ndarray]:
        masks: List[bool] = []
        minigrid_ego_image = []
        minigrid_mission = []
        expert_actions = []
        q = self.rollout_queues[rollout_ind]
        while len(masks) != self.rollout_len:
            if q.empty():
                if not self.add_data_to_rollout_queue(q):
                    raise StopIteration()

            obs, expert_action, _, is_first_obs = cast(
                Tuple[
                    Dict[str, Union[np.array, int, str]],
                    MiniGridEnv.Actions,
                    bool,
                    bool,
                ],
                q.get_nowait(),
            )

            masks.append(not is_first_obs)
            minigrid_ego_image.append(obs["image"])
            if self.minigrid_mission_sensor is not None:
                # noinspection PyTypeChecker
                minigrid_mission.append(
                    self.minigrid_mission_sensor.get_observation(
                        env=None, task=None, minigrid_output_obs=obs
                    )
                )
            expert_actions.append([expert_action])

        to_return = {
            "masks": np.array(masks, dtype=np.float32).reshape(
                (self.rollout_len, 1, 1)  # steps x agent x mask
            ),
            "minigrid_ego_image": np.stack(
                minigrid_ego_image, axis=0
            ),  # steps x height x width x channels
            "expert_action": np.array(expert_actions, dtype=np.int64).reshape(
                (self.rollout_len, 1, -1)  # steps x agent x action
            ),
        }
        if self.minigrid_mission_sensor is not None:
            to_return["minigrid_mission"] = np.stack(
                minigrid_mission, axis=0
            )  # steps x mission_dims
        return to_return

    def __next__(self) -> Dict[str, torch.Tensor]:
        all_data = defaultdict(lambda: [])
        for rollout_ind in range(self.nrollouts):
            data_for_ind = self.get_data_for_rollout_ind(rollout_ind=rollout_ind)
            for key in data_for_ind:
                all_data[key].append(data_for_ind[key])
        return {
            key: torch.from_numpy(np.stack(all_data[key], axis=1))  # new sampler dim
            for key in all_data
        }


def create_minigrid_offpolicy_data_iterator(
    path: str,
    nrollouts: int,
    rollout_len: int,
    instr_len: Optional[int],
    restrict_max_steps_in_dataset: Optional[int] = None,
) -> ExpertTrajectoryIterator:
    path = os.path.abspath(path)

    if path not in _DATASET_CACHE:
        get_logger().info(
            "Loading minigrid dataset from {} for first time...".format(path)
        )
        _DATASET_CACHE[path] = babyai.utils.load_demos(path)
        assert _DATASET_CACHE[path] is not None and len(_DATASET_CACHE[path]) != 0
        get_logger().info(
            "Loading minigrid dataset complete, it contains {} trajectories".format(
                len(_DATASET_CACHE[path])
            )
        )
    return ExpertTrajectoryIterator(
        data=_DATASET_CACHE[path],
        nrollouts=nrollouts,
        rollout_len=rollout_len,
        instr_len=instr_len,
        restrict_max_steps_in_dataset=restrict_max_steps_in_dataset,
    )
