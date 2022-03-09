import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.grouped_action_imitation import (
    GroupedActionImitation,
)
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
)
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from allenact_plugins.ithor_plugin.ithor_sensors import TakeEndActionThorNavSensor
from allenact_plugins.robothor_plugin import robothor_constants
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.mixins import ResNetPreprocessGRUActorCriticMixin


class ObjectNavRoboThorResNet18GRURGBPPOExperimentConfig(ObjectNavRoboThorBaseConfig):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    SENSORS = (  # type:ignore
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
        TakeEndActionThorNavSensor(
            nactions=len(ObjectNavTask.class_action_names()), uuid="expert_group_action"
        ),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.preprocessing_and_model = ResNetPreprocessGRUActorCriticMixin(
            sensors=self.SENSORS,
            resnet_type="RN18",
            screen_size=self.SCREEN_SIZE,
            goal_sensor_type=GoalObjectTypeThorSensor,
        )

    def preprocessors(self):
        return self.preprocessing_and_model.preprocessors()

    def create_model(self, **kwargs):
        return self.preprocessing_and_model.create_model(
            num_actions=self.ACTION_SPACE.n, **kwargs
        )

    def training_pipeline(self, **kwargs):
        ppo_steps = int(300000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128
        save_interval = 5000000
        log_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5

        action_strs = ObjectNavTask.class_action_names()
        non_end_action_inds_set = {
            i for i, a in enumerate(action_strs) if a != robothor_constants.END
        }
        end_action_ind_set = {action_strs.index(robothor_constants.END)}

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={
                "ppo_loss": PPO(**PPOConfig),
                "grouped_action_imitation": GroupedActionImitation(
                    nactions=len(ObjectNavTask.class_action_names()),
                    action_groups=[non_end_action_inds_set, end_action_ind_set],
                ),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss", "grouped_action_imitation"],
                    max_stage_steps=ppo_steps,
                )
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    def tag(self):
        return "ObjectNav-RoboTHOR-RGB-ResNet18GRU-DDPPOAndGBC"
