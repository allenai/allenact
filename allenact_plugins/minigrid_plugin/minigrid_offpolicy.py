import math
import queue
import random
from collections import defaultdict
from typing import Dict, Tuple, Any, cast, List, Union, Optional

import babyai
import blosc
import numpy as np
import pickle5 as pickle
import torch
from gym_minigrid.minigrid import MiniGridEnv

from allenact.algorithms.offpolicy_sync.losses.abstract_offpolicy_loss import Memory
from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.algorithms.onpolicy_sync.storage import (
    ExperienceStorage,
    StreamingStorageMixin,
)
from allenact.base_abstractions.misc import GenericAbstractLoss, LossOutput, ModelType
from allenact.utils.misc_utils import partition_limits
from allenact.utils.system import get_logger
from allenact_plugins.minigrid_plugin.minigrid_sensors import MiniGridMissionSensor

_DATASET_CACHE: Dict[str, Any] = {}


class MiniGridOffPolicyExpertCELoss(GenericAbstractLoss):
    def __init__(self, total_episodes_in_epoch: Optional[int] = None):
        super().__init__()
        self.total_episodes_in_epoch = total_episodes_in_epoch

    def loss(  # type: ignore
        self,
        *,  # No positional arguments
        model: ModelType,
        batch: ObservationType,
        batch_memory: Memory,
        stream_memory: Memory,
    ) -> LossOutput:
        rollout_len, nrollouts = cast(torch.Tensor, batch["minigrid_ego_image"]).shape[
            :2
        ]

        # Initialize Memory if empty
        if len(stream_memory) == 0:
            spec = model.recurrent_memory_specification
            for key in spec:
                dims_template, dtype = spec[key]
                # get sampler_dim and all_dims from dims_template (and nrollouts)

                dim_names = [d[0] for d in dims_template]
                sampler_dim = dim_names.index("sampler")

                all_dims = [d[1] for d in dims_template]
                all_dims[sampler_dim] = nrollouts

                stream_memory.check_append(
                    key=key,
                    tensor=torch.zeros(
                        *all_dims,
                        dtype=dtype,
                        device=cast(torch.Tensor, batch["minigrid_ego_image"]).device,
                    ),
                    sampler_dim=sampler_dim,
                )

        # Forward data (through the actor and critic)
        ac_out, stream_memory = model.forward(
            observations=batch,
            memory=stream_memory,
            prev_actions=None,  # type:ignore
            masks=cast(torch.FloatTensor, batch["masks"]),
        )

        # Compute the loss from the actor's output and expert action
        expert_ce_loss = -ac_out.distributions.log_prob(batch["expert_action"]).mean()

        info = {"expert_ce": expert_ce_loss.item()}

        return LossOutput(
            value=expert_ce_loss,
            info=info,
            per_epoch_info={},
            batch_memory=batch_memory,
            stream_memory=stream_memory,
            bsize=rollout_len * nrollouts,
        )


def transform_demos(demos):
    # A modified version of babyai.utils.demos.transform_demos
    # where we use pickle 5 instead of standard pickle
    new_demos = []
    for demo in demos:
        new_demo = []

        mission = demo[0]
        all_images = demo[1]
        directions = demo[2]
        actions = demo[3]

        # First decompress the pickle
        pickled_array = blosc.blosc_extension.decompress(all_images, False)
        # ... and unpickle
        all_images = pickle.loads(pickled_array)

        n_observations = all_images.shape[0]
        assert (
            len(directions) == len(actions) == n_observations
        ), "error transforming demos"
        for i in range(n_observations):
            obs = {
                "image": all_images[i],
                "direction": directions[i],
                "mission": mission,
            }
            action = actions[i]
            done = i == n_observations - 1
            new_demo.append((obs, action, done))
        new_demos.append(new_demo)
    return new_demos


class MiniGridExpertTrajectoryStorage(ExperienceStorage, StreamingStorageMixin):
    def __init__(
        self,
        data_path: str,
        num_samplers: int,
        rollout_len: int,
        instr_len: Optional[int],
        restrict_max_steps_in_dataset: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super(MiniGridExpertTrajectoryStorage, self).__init__()
        self.data_path = data_path
        self._data: Optional[
            List[Tuple[str, bytes, List[int], MiniGridEnv.Actions]]
        ] = None
        self.restrict_max_steps_in_dataset = restrict_max_steps_in_dataset

        self.original_num_samplers = num_samplers
        self.num_samplers = num_samplers

        self.rollout_len = rollout_len
        self.instr_len = instr_len

        self.current_worker = 0
        self.num_workers = 1

        self.minigrid_mission_sensor: Optional[MiniGridMissionSensor] = None
        if instr_len is not None:
            self.minigrid_mission_sensor = MiniGridMissionSensor(instr_len)

        self.rollout_queues = []
        self._remaining_inds = []
        self.sampler_to_num_steps_in_queue = []
        self._total_experiences = 0

        self.device = device

    @property
    def data(self) -> List[Tuple[str, bytes, List[int], MiniGridEnv.Actions]]:
        if self._data is None:
            if self.data_path not in _DATASET_CACHE:
                get_logger().info(
                    f"Loading minigrid dataset from {self.data_path} for first time..."
                )
                _DATASET_CACHE[self.data_path] = babyai.utils.load_demos(self.data_path)
                assert (
                    _DATASET_CACHE[self.data_path] is not None
                    and len(_DATASET_CACHE[self.data_path]) != 0
                )
                get_logger().info(
                    "Loading minigrid dataset complete, it contains {} trajectories".format(
                        len(_DATASET_CACHE[self.data_path])
                    )
                )
            self._data = _DATASET_CACHE[self.data_path]

            if self.restrict_max_steps_in_dataset is not None:
                restricted_data = []
                cur_len = 0
                for i, d in enumerate(self._data):
                    if cur_len >= self.restrict_max_steps_in_dataset:
                        break
                    restricted_data.append(d)
                    cur_len += len(d[2])
                self._data = restricted_data

            parts = partition_limits(len(self._data), self.num_workers)
            self._data = self._data[
                parts[self.current_worker] : parts[self.current_worker + 1]
            ]

            self.rollout_queues = [queue.Queue() for _ in range(self.num_samplers)]
            self.sampler_to_num_steps_in_queue = [0 for _ in range(self.num_samplers)]
            for it, q in enumerate(self.rollout_queues):
                self._fill_rollout_queue(q, it)

        return self._data

    def set_partition(self, index: int, num_parts: int):
        self.current_worker = index
        self.num_workers = num_parts

        self.num_samplers = int(math.ceil(self.original_num_samplers / num_parts))

        self._data = None

        for q in self.rollout_queues:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
        self.rollout_queues = []

    def initialize(self, *, observations: ObservationType, **kwargs):
        self.reset_stream()
        assert len(self.data) != 0

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
        pass

    def to(self, device: torch.device):
        self.device = device

    @property
    def total_experiences(self) -> int:
        return self._total_experiences

    def reset_stream(self):
        self.set_partition(index=self.current_worker, num_parts=self.num_workers)

    def empty(self) -> bool:
        return False

    def _get_next_ind(self):
        if len(self._remaining_inds) == 0:
            self._remaining_inds = list(range(len(self.data)))
            random.shuffle(self._remaining_inds)
        return self._remaining_inds.pop()

    def _fill_rollout_queue(self, q: queue.Queue, sampler: int):
        assert q.empty()

        while self.sampler_to_num_steps_in_queue[sampler] < self.rollout_len:
            next_ind = self._get_next_ind()

            for i, step in enumerate(transform_demos([self.data[next_ind]])[0]):
                q.put((*step, i == 0))
                self.sampler_to_num_steps_in_queue[sampler] += 1

        return True

    def get_data_for_rollout_ind(self, sampler_ind: int) -> Dict[str, np.ndarray]:
        masks: List[bool] = []
        minigrid_ego_image = []
        minigrid_mission = []
        expert_actions = []
        q = self.rollout_queues[sampler_ind]
        while len(masks) != self.rollout_len:
            if q.empty():
                assert self.sampler_to_num_steps_in_queue[sampler_ind] == 0
                self._fill_rollout_queue(q, sampler_ind)

            obs, expert_action, _, is_first_obs = cast(
                Tuple[
                    Dict[str, Union[np.array, int, str]],
                    MiniGridEnv.Actions,
                    bool,
                    bool,
                ],
                q.get_nowait(),
            )
            self.sampler_to_num_steps_in_queue[sampler_ind] -= 1

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
            "masks": torch.tensor(masks, device=self.device, dtype=torch.float32).view(
                self.rollout_len, 1  # steps x mask
            ),
            "minigrid_ego_image": torch.stack(
                [torch.tensor(img, device=self.device) for img in minigrid_ego_image],
                dim=0,
            ),  # steps x height x width x channels
            "expert_action": torch.tensor(
                expert_actions, device=self.device, dtype=torch.int64
            ).view(
                self.rollout_len  # steps
            ),
        }
        if self.minigrid_mission_sensor is not None:
            to_return["minigrid_mission"] = torch.stack(
                [torch.tensor(m, device=self.device) for m in minigrid_mission], dim=0
            )  # steps x mission_dims
        return to_return

    def next_batch(self) -> Dict[str, torch.Tensor]:
        all_data = defaultdict(lambda: [])
        for rollout_ind in range(self.num_samplers):
            data_for_ind = self.get_data_for_rollout_ind(sampler_ind=rollout_ind)
            for key in data_for_ind:
                all_data[key].append(data_for_ind[key])

        self._total_experiences += self.num_samplers * self.rollout_len
        return {
            key: torch.stack(all_data[key], dim=1,)  # new sampler dim
            for key in all_data
        }
