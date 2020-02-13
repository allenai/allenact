import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from .object_nav_thor_ppo import ObjectNavThorPPOExperimentConfig
from onpolicy_sync.losses import PPO
from onpolicy_sync.losses.imitation import Imitation
from onpolicy_sync.losses.ppo import PPOConfig
from rl_ai2thor.ai2thor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from rl_base.sensor import ExpertActionSensor
from utils.experiment_utils import LinearDecay, Builder, PipelineStage, TrainingPipeline


class ObjectNavThorDAggerPPOExperimentConfig(ObjectNavThorPPOExperimentConfig):
    """An object navigation experiment in THOR.

    Training with imitation learning (DAgger) and then PPO. Extends
    ObjectNavThorPPOExperimentConfig, see that config for more details.
    """

    SCREEN_SIZE = 224

    # Easy setting
    EASY = True
    OBJECT_TYPES = sorted(["Tomato"])
    TRAIN_SCENES = ["FloorPlan1_physics"]
    VALID_SCENES = ["FloorPlan1_physics"]
    TEST_SCENES = ["FloorPlan1_physics"]

    # Hard setting
    # EASY = False
    # OBJECT_TYPES = sorted(["Cup", "Television", "Tomato"])
    # TRAIN_SCENES = ["FloorPlan{}".format(i) for i in range(1, 21)]
    # VALID_SCENES = ["FloorPlan{}_physics".format(i) for i in range(21, 26)]
    # TEST_SCENES = ["FloorPlan{}_physics".format(i) for i in range(26, 31)]

    SENSORS = [
        RGBSensorThor(
            {
                "height": SCREEN_SIZE,
                "width": SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        GoalObjectTypeThorSensor({"object_types": OBJECT_TYPES}),
        ExpertActionSensor({"nactions": 6}),
    ]

    @classmethod
    def tag(cls):
        return "ObjectNavThorDAggerPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        dagger_steps = int(3e4) if cls.EASY else int(1e5)
        ppo_steps = int(3e4) if cls.EASY else int(1e7)
        lr = 2.5e-4
        num_mini_batch = 1 if not torch.cuda.is_available() else 6
        update_repeats = 4
        num_steps = 128
        log_interval = cls.MAX_STEPS * 10  # Log every 10 max length tasks
        save_interval = 10000 if cls.EASY else 500000
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            log_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={
                "imitation_loss": Builder(Imitation,),
                "ppo_loss": Builder(PPO, default=PPOConfig,),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=0.0, steps=dagger_steps,
                    ),
                    end_criterion=dagger_steps,
                ),
                PipelineStage(
                    loss_names=["ppo_loss", "imitation_loss"], end_criterion=ppo_steps
                ),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=dagger_steps + ppo_steps)}
            ),
        )
