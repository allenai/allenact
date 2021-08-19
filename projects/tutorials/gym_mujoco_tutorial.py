from typing import Dict, Optional, List, Any, cast

import gym
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses.ppo import PPO

from allenact.algorithms.onpolicy_sync.losses.a2cacktr import A2C
from allenact.algorithms.onpolicy_sync.losses.a2cacktr import A2CConfig

from allenact.base_abstractions.experiment_config import ExperimentConfig, TaskSampler
from allenact.base_abstractions.sensor import SensorSuite
from allenact_plugins.gym_plugin.gym_models import MemorylessActorCritic
from allenact_plugins.gym_plugin.gym_sensors import GymMuJoCoSensor

from allenact_plugins.gym_plugin.gym_tasks import GymTaskSampler
from allenact.utils.experiment_utils import (
    TrainingPipeline,
    Builder,
    PipelineStage,
    LinearDecay,
)
from allenact.utils.viz_utils import VizSuite, AgentViewViz

"""
For Test MuJoCo environment:
    'InvertedPendulum-v2',
    'Ant-v2',
    'InvertedDoublePendulum-v2',
    'Humanoid-v2',
    'Reacher-v2',
    'Hopper-v2',
    'HalfCheetah-v2',
    'Swimmer-v2',
    'Walker2d-v2',

For Test Robotics Environment:
    'HandManipulateBlock-v0',
    'FetchPickAndPlace-v1',
    'FetchReach-v1',
    'FetchSlide-v1',
    'HandManipulateEgg-v0',
    'HandReach-v0',
    'HandManipulatePen-v0'
"""


ENV = {"id": "gym_mujoco_data", "env": "Humanoid-v2", "loss": "ppo"}


class HandManipulateTutorialExperimentConfig(ExperimentConfig):
    @classmethod
    def tag(cls) -> str:
        return "Gym_extra_Tutorial"

    SENSORS = [
        GymMuJoCoSensor(ENV["env"], uuid=ENV["id"]),
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
        # action space for gym MoJoCo
        if ENV["env"] == "InvertedPendulum-v2":
            action_space = gym.spaces.Box(-3.0, 3.0, (1,), "float32")
        elif ENV["env"] == "Ant-v2":
            action_space = gym.spaces.Box(-3.0, 3.0, (8,), "float32")
        elif ENV["env"] in ["Humanoid-v2", "HumanoidStandup-v2"]:
            action_space = gym.spaces.Box(
                -0.4000000059604645, 0.4000000059604645, (17,), "float32"
            )
        elif ENV["env"] == "InvertedDoublePendulum-v2":
            action_space = gym.spaces.Box(-1.0, 1.0, (1,), "float32")
        elif ENV["env"] in ["Reacher-v2", "Swimmer-v2"]:
            action_space = gym.spaces.Box(-1.0, 1.0, (2,), "float32")
        elif ENV["env"] == "Hopper-v2":
            action_space = gym.spaces.Box(-1.0, 1.0, (3,), "float32")
        elif ENV["env"] in ["HalfCheetah-v2", "Walker2d-v2"]:
            action_space = gym.spaces.Box(-1.0, 1.0, (6,), "float32")
        # TODO action space for gym Robotics
        elif ENV["env"] == "HandManipulateBlock-v0":
            action_space = gym.spaces.Box(-1.0, 1.0, (20,), "float32")
        else:
            raise NotImplementedError
        return MemorylessActorCritic(
            input_uuid=ENV["id"],
            action_space=action_space,  # specific action_space
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            action_std=0.5,
        )

    """
    Task samplers
    We use an available `TaskSampler` implementation for `gym` environments that allows to sample GymTasks
    GymTaskSampler. Even though it is possible to let the task sampler instantiate the proper sensor for the 
    chosen task name (by passing `None`), we use the sensors we created above, which contain a custom identifier 
    for the actual observation space (`gym_hand(mujoco)_data`) also used by the model.
    """

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return GymTaskSampler(gym_env_type=ENV["env"], **kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(
            process_ind=process_ind, mode="train", seeds=seeds
        )

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(
            process_ind=process_ind, mode="valid", seeds=seeds
        )

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="test", seeds=seeds)

    """
    The task sampler samples random tasks for ever, while,
    during testing (or validation), we sample a fixed number of tasks.
    """

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
            max_tasks = 2

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
            gym_env_types=[ENV["env"]],
            sensors=self.SENSORS,  # sensors used to return observations to the agent
            max_tasks=max_tasks,  # see above
            task_seeds_list=task_seeds_list,  # see above
            deterministic_sampling=deterministic_sampling,  # see above
            seed=seeds[process_ind],
        )

    """
    Note that we just sample 3 tasks for validation and testing in this case, which suffice to illustrate the model's
    success.
    """

    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> Dict[str, Any]:
        visualizer = None
        if mode == "test":
            visualizer = VizSuite(
                mode=mode,
                video_viz=AgentViewViz(
                    label="episode_vid",
                    max_clip_length=400,
                    vector_task_source=("render", {"mode": "rgb_array"}),
                    fps=30,
                ),
            )
        return {
            "nprocesses": 8 if mode == "train" else 1,  # rollout
            "devices": [],
            "visualizer": visualizer,
        }

    """
    Training pipeline
    The last definition is the training pipeline. In this case, we use a PPO/A2C stage with linearly decaying learning 
    rate and 80 single-batch update repeats per rollout:
    """

    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        lr = 1e-4  # 1e4
        if ENV["loss"] == "ppo":
            ppo_steps = int(1e8)
            return TrainingPipeline(
                named_losses=dict(
                    ppo_loss=PPO(
                        clip_param=0.1,  # 0.2
                        value_loss_coef=0.5,
                        entropy_coef=0.0,
                    ),
                ),  # type:ignore
                pipeline_stages=[
                    PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps),
                ],
                optimizer_builder=Builder(
                    cast(optim.Optimizer, optim.Adam), dict(lr=lr)
                ),
                num_mini_batch=4,  # 64
                update_repeats=10,  # 10
                max_grad_norm=0.5,  # 0.5
                num_steps=2048,  # 2048
                gamma=0.99,
                use_gae=True,
                gae_lambda=0.95,
                advance_scene_rollout_period=None,
                save_interval=500000,
                metric_accumulate_interval=50000,
                lr_scheduler_builder=Builder(
                    LambdaLR,
                    {
                        "lr_lambda": LinearDecay(steps=ppo_steps, startp=1, endp=1)
                    },  # startp=1, endp=1 for constant
                ),
            )
        elif ENV["loss"] == "a2c":
            a2c_steps = int(1e6)
            return TrainingPipeline(
                named_losses={"a2c_loss": A2C(**A2CConfig)},  # type:ignore
                pipeline_stages=[
                    PipelineStage(loss_names=["a2c_loss"], max_stage_steps=a2c_steps),
                ],
                optimizer_builder=Builder(
                    cast(optim.Optimizer, optim.Adam), dict(lr=lr)
                ),
                num_mini_batch=1,
                update_repeats=80,
                max_grad_norm=100,
                num_steps=2000,
                gamma=0.99,
                use_gae=False,
                gae_lambda=0.95,
                advance_scene_rollout_period=None,
                save_interval=200000,
                metric_accumulate_interval=50000,
                lr_scheduler_builder=Builder(
                    LambdaLR,
                    {"lr_lambda": LinearDecay(steps=a2c_steps)},  # type:ignore
                ),
            )
        else:
            raise NotImplementedError
