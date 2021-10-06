from typing import Dict, Optional, List, Any, cast
import os

import gym
from gym_minigrid.envs import EmptyRandomEnv5x5
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses.ppo import PPO, PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig, TaskSampler
from allenact.base_abstractions.sensor import SensorSuite, ExpertActionSensor
from allenact.utils.experiment_utils import (
    TrainingPipeline,
    Builder,
    PipelineStage,
    LinearDecay,
)
from allenact_plugins.minigrid_plugin.minigrid_sensors import EgocentricMiniGridSensor
from allenact_plugins.minigrid_plugin.minigrid_tasks import MiniGridTaskSampler
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from tempfile import mkdtemp
from allenact.algorithms.onpolicy_sync.runner import OnPolicyRunner
from projects.tutorials.minigrid_tutorial_conds import (
    ConditionedMiniGridSimpleConvRNN,
    ConditionedMiniGridTask,
)


class MiniGridCondTestExperimentConfig(ExperimentConfig):
    @classmethod
    def tag(cls) -> str:
        return "MiniGridCondTest"

    SENSORS = [
        EgocentricMiniGridSensor(agent_view_size=5, view_channels=3),
        ExpertActionSensor(
            action_space=gym.spaces.Dict(
                higher=gym.spaces.Discrete(2), lower=gym.spaces.Discrete(2)
            )
        ),
    ]

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ConditionedMiniGridSimpleConvRNN(
            action_space=gym.spaces.Dict(
                higher=gym.spaces.Discrete(2), lower=gym.spaces.Discrete(2)
            ),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            num_objects=cls.SENSORS[0].num_objects,
            num_colors=cls.SENSORS[0].num_colors,
            num_states=cls.SENSORS[0].num_states,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return MiniGridTaskSampler(**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="train")

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="valid")

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(process_ind=process_ind, mode="test")

    def _get_sampler_args(self, process_ind: int, mode: str) -> Dict[str, Any]:
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
            max_tasks = 20 + 20 * (
                mode == "test"
            )  # 20 tasks for valid, 40 for test (per sampler)

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
            max_tasks=max_tasks,  # see above
            env_class=self.make_env,  # builder for third-party environment (defined below)
            sensors=self.SENSORS,  # sensors used to return observations to the agent
            env_info=dict(),  # parameters for environment builder (none for now)
            task_seeds_list=task_seeds_list,  # see above
            deterministic_sampling=deterministic_sampling,  # see above
            task_class=ConditionedMiniGridTask,
        )

    @staticmethod
    def make_env(*args, **kwargs):
        return EmptyRandomEnv5x5()

    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> Dict[str, Any]:
        return {
            "nprocesses": 4 if mode == "train" else 1,
            "devices": [],
        }

    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        ppo_steps = int(512)
        return TrainingPipeline(
            named_losses=dict(
                imitation_loss=Imitation(
                    cls.SENSORS[1]
                ),  # 0 is Minigrid, 1 is ExpertActionSensor
                ppo_loss=PPO(**PPOConfig, entropy_method_name="conditional_entropy"),
            ),  # type:ignore
            pipeline_stages=[
                PipelineStage(
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=0.0, steps=ppo_steps // 2,
                    ),
                    loss_names=["imitation_loss", "ppo_loss"],
                    max_stage_steps=ppo_steps,
                )
            ],
            optimizer_builder=Builder(cast(optim.Optimizer, optim.Adam), dict(lr=1e-4)),
            num_mini_batch=4,
            update_repeats=3,
            max_grad_norm=0.5,
            num_steps=16,
            gamma=0.99,
            use_gae=True,
            gae_lambda=0.95,
            advance_scene_rollout_period=None,
            save_interval=10000,
            metric_accumulate_interval=1,
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}  # type:ignore
            ),
        )


class TestMiniGridCond:
    def test_train(self, tmpdir):
        cfg = MiniGridCondTestExperimentConfig()
        train_runner = OnPolicyRunner(
            config=cfg,
            output_dir=tmpdir,
            loaded_config_src_files=None,
            seed=12345,
            mode="train",
            deterministic_cudnn=False,
            deterministic_agents=False,
            extra_tag="",
            disable_tensorboard=True,
            disable_config_saving=True,
        )
        start_time_str, valid_results = train_runner.start_train(
            checkpoint=None,
            restart_pipeline=False,
            max_sampler_processes_per_worker=1,
            collect_valid_results=True,
        )
        assert len(valid_results) > 0

        test_runner = OnPolicyRunner(
            config=cfg,
            output_dir=tmpdir,
            loaded_config_src_files=None,
            seed=12345,
            mode="test",
            deterministic_cudnn=False,
            deterministic_agents=False,
            extra_tag="",
            disable_tensorboard=True,
            disable_config_saving=True,
        )
        test_results = test_runner.start_test(
            checkpoint_path_dir_or_pattern=os.path.join(
                tmpdir, "checkpoints", "**", start_time_str, "*.pt"
            ),
            max_sampler_processes_per_worker=1,
            inference_expert=True,
        )
        assert test_results[-1]["ep_length"] < 4


if __name__ == "__main__":
    TestMiniGridCond().test_train(mkdtemp())  # type:ignore
