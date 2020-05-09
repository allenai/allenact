from typing import Dict, Any, List, Optional

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import habitat
from onpolicy_sync.losses.ppo import PPOConfig
from models.object_nav_models import ObjectNavActorCriticTrainResNet50RNN
from onpolicy_sync.losses import PPO
from rl_base.experiment_config import ExperimentConfig
from rl_base.sensor import SensorSuite
from rl_base.task import TaskSampler
from rl_base.preprocessor import ObservationSet
from rl_habitat.habitat_tasks import ObjectNavTask
from rl_habitat.habitat_task_samplers import ObjectNavTaskSampler
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay
from utils.viz_utils import SimpleViz, AgentViewViz, TrajectoryViz, ActorViz


class ObjectNavHabitatDDPPOBaseExperimentConfig(ExperimentConfig):
    """A Point Navigation experiment configuraqtion in Habitat"""

    TRAIN_SCENES = "habitat/habitat-api/data/datasets/objectnav/mp3d/v0/train_rooms/train.json.gz"
    VALID_SCENES = "habitat/habitat-api/data/datasets/objectnav/mp3d/v0/val_rooms/val.json.gz"

    SCREEN_SIZE = 256
    MAX_STEPS = 2000
    DISTANCE_TO_GOAL = 0.5

    NUM_PROCESSES = 24

    CONFIG = habitat.get_config('configs/mp3d.yaml')
    CONFIG.defrost()
    CONFIG.NUM_PROCESSES = NUM_PROCESSES if torch.cuda.is_available() else 1
    CONFIG.SIMULATOR_GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
    CONFIG.DATASET.TYPE = 'ObjectNav-v1'
    CONFIG.DATASET.SCENES_DIR = 'habitat/habitat-api/data/scene_datasets/'
    CONFIG.DATASET.DATA_PATH = TRAIN_SCENES
    CONFIG.SIMULATOR.AGENT_0.HEIGHT = 0.88
    CONFIG.SIMULATOR.AGENT_0.RADIUS = 0.18
    CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = False

    CONFIG.SIMULATOR.TURN_ANGLE = 10
    CONFIG.SIMULATOR.TILT_ANGLE = 30
    CONFIG.SIMULATOR.ACTION_SPACE_CONFIG = "v1"
    CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = MAX_STEPS

    CONFIG.TASK.TYPE = 'ObjectNav-v1'
    CONFIG.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
    CONFIG.TASK.SUCCESS_DISTANCE = DISTANCE_TO_GOAL
    CONFIG.TASK.SENSORS = ['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']
    CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.GOAL_FORMAT = "POLAR"
    CONFIG.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.DIMENSIONALITY = 2
    CONFIG.TASK.GOAL_SENSOR_UUID = 'objectgoal'
    CONFIG.TASK.MEASUREMENTS = ['DISTANCE_TO_GOAL', 'SPL']
    CONFIG.TASK.SPL.TYPE = 'SPL'
    CONFIG.TASK.SPL.DISTANCE_TO = 'VIEW_POINTS'  # "POINT"
    CONFIG.TASK.SPL.SUCCESS_DISTANCE = DISTANCE_TO_GOAL
    CONFIG.TASK.DISTANCE_TO_GOAL.DISTANCE_TO = 'VIEW_POINTS'  # 'POINT'

    CONFIG.MODE = 'train'
    VISUALIZATION_IDS = ['x8F5xyUWy9e_46', 'zsNo4HB9uLZ_64']

    @classmethod
    def tag(cls):
        return "ObjectNav"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = 2.5e8
        lr = 2.5e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128
        save_interval = 5000000
        log_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            log_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"ppo_loss": Builder(PPO, kwargs={"use_clipped_value_loss": True}, default=PPOConfig,)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=None,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], end_criterion=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def evaluation_params(cls, **kwargs):
        nprocesses = 1
        gpu_ids = [] if not torch.cuda.is_available() else [7]
        res = cls.training_pipeline()
        del res["pipeline"]
        del res["optimizer"]
        res["nprocesses"] = nprocesses
        res["gpu_ids"] = gpu_ids
        return res

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            nprocesses = 1 if not torch.cuda.is_available() else [3, 3, 3, 3, 3, 3, 3, 3]
            gpu_ids = [] if not torch.cuda.is_available() else self.CONFIG.SIMULATOR_GPU_IDS
            render_video = False
        elif mode == "valid":
            nprocesses = 1
            if not torch.cuda.is_available():
                gpu_ids = []
            else:
                gpu_ids = [7]
            render_video = False
        elif mode == "test":
            nprocesses = 1
            if not torch.cuda.is_available():
                gpu_ids = []
            else:
                gpu_ids = [1]
            render_video = True
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        observation_set = Builder(ObservationSet, kwargs=dict(
            source_ids=self.OBSERVATIONS, all_preprocessors=self.PREPROCESSORS, all_sensors=self.SENSORS
        )) if mode == 'train' or nprocesses > 0 else None

        if mode == "valid":
            visualizer = Builder(SimpleViz, dict(
                episode_ids=self.VISUALIZATION_IDS,  # which episodes to log, List[str] or List[List[str]] to split into viz groups
                mode="valid",  # or valid
                v1=Builder(TrajectoryViz, dict()),  # trajectory
                v2=Builder(AgentViewViz, dict(max_video_length=100, episode_ids=self.VISUALIZATION_IDS)),  # first person videos
                v3=Builder(ActorViz, dict()),  # action probs
                # v4=Builder(TensorViz1D, dict()),  # visualize 1D tensor (time) from rollout
                # v5=Builder(TensorViz1D, dict(rollout_source=("masks"))),
                # v6=Builder(TensorViz2D, dict()),
                # visualize 2D tensor (time + another dim, e.g. hidden states) from rollout
                path_to_id=('task_info', 'episode_id')
            ))
            return {
                "nprocesses": nprocesses,
                "gpu_ids": gpu_ids,
                "observation_set": observation_set,
                "render_video": render_video,
                "visualizer": visualizer,
            }

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "observation_set": observation_set,
            "render_video": render_video,
        }

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjectNavActorCriticTrainResNet50RNN(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            hidden_size=512,
            object_type_embedding_dim=8,
            trainable_masked_hidden_state=False,
            num_rnn_layers=1,
            rnn_type='GRU'
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavTaskSampler(**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.TRAIN_CONFIGS[process_ind]
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
        }

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.CONFIG.clone()
        config.defrost()
        config.DATASET.DATA_PATH = self.VALID_SCENES
        config.MODE = 'validate'
        config.SIMULATOR_GPU_IDS = [0]
        config.freeze()
        return {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(len(ObjectNavTask.action_names())),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
            "id": process_ind
        }
