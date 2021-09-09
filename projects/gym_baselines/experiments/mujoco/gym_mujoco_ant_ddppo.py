from typing import Dict, List, Any

import gym
import torch.nn as nn

from allenact.base_abstractions.experiment_config import TaskSampler
from allenact.base_abstractions.sensor import SensorSuite
from allenact_plugins.gym_plugin.gym_models import MemorylessActorCritic
from allenact_plugins.gym_plugin.gym_sensors import GymMuJoCoSensor

from allenact_plugins.gym_plugin.gym_tasks import GymTaskSampler

from projects.gym_baselines.experiments.gym_mujoco_ddppo import GymMuJoCoPPOConfig


class GymMuJoCoAntConfig(GymMuJoCoPPOConfig):

    SENSORS = [
        GymMuJoCoSensor(gym_env_name="Ant-v2", uuid="gym_mujoco_data"),
    ]

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        """We define our `ActorCriticModel` agent using a lightweight
        implementation with separate MLPs for actors and critic,
        MemorylessActorCritic.

        Since this is a model for continuous control, note that the
        superclass of our model is `ActorCriticModel[GaussianDistr]`
        instead of `ActorCriticModel[CategoricalDistr]`, since we'll use
        a Gaussian distribution to sample actions.
        """
        action_space = gym.spaces.Box(-3.0, 3.0, (8,), "float32")
        return MemorylessActorCritic(
            input_uuid="gym_mujoco_data",
            action_space=action_space,  # specific action_space
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            action_std=0.5,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return GymTaskSampler(gym_env_type="Ant-v2", **kwargs)

    def _get_sampler_args(
        self, process_ind: int, mode: str, seeds: List[int]
    ) -> Dict[str, Any]:
        """Generate initialization arguments for train, valid, and test
        TaskSamplers.

        # Parameters
        process_ind : index of the current task sampler
        mode:  one of `train`, `valid`, or `test`
        """
        if mode == "train":
            max_tasks = None  # infinite training tasks
            task_seeds_list = None  # no predefined random seeds for training
            deterministic_sampling = False  # randomly sample tasks in training
        else:
            max_tasks = 4

            # one seed for each task to sample:
            # - ensures different seeds for each sampler, and
            # - ensures a deterministic set of sampled tasks.
            task_seeds_list = list(
                range(process_ind * max_tasks, (process_ind + 1) * max_tasks)
            )

            deterministic_sampling = (
                True  # deterministically sample task in validation/testing
            )

        return dict(
            gym_env_types=["Ant-v2"],
            sensors=self.SENSORS,  # sensors used to return observations to the agent
            max_tasks=max_tasks,  # see above
            task_seeds_list=task_seeds_list,  # see above
            deterministic_sampling=deterministic_sampling,  # see above
            seed=seeds[process_ind],
        )

    @classmethod
    def tag(cls) -> str:
        return "Gym-MuJoCo-Ant-v2-PPO"
