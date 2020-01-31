import torch
import torch.optim as optim

from experiments.object_nav_thor_ppo import ObjectNavThorPPOExperimentConfig
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

    # Easy setting
    OBJECT_TYPES = sorted(["Tomato"])
    TRAIN_SCENES = ["FloorPlan1_physics"]
    VALID_SCENES = ["FloorPlan1_physics"]
    TEST_SCENES = ["FloorPlan1_physics"]

    # Hard setting
    # OBJECT_TYPES = sorted(["Cup", "Television", "Tomato"])
    # TRAIN_SCENES = ["FloorPlan{}".format(i) for i in range(1, 21)]
    # VALID_SCENES = ["FloorPlan{}_physics".format(i) for i in range(21, 26)]
    # TEST_SCENES = ["FloorPlan{}_physics".format(i) for i in range(26, 31)]

    SCREEN_SIZE = 224

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
        dagger_steps = int(3e4)
        ppo_steps = int(3e4)
        lr = 2.5e-4
        num_mini_batch = 6 if not torch.cuda.is_available() else 30
        update_repeats = 3
        num_steps = 128
        log_interval = cls.MAX_STEPS * 10  # Log every 10 max length tasks
        save_interval = 10000  # Save every 10000 steps (approximately)
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            log_interval=log_interval,
            optimizer=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            num_steps=num_steps,
            named_losses={
                "imitation_loss": Builder(Imitation,),
                "ppo_loss": Builder(PPO, default=PPOConfig,),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
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
        )
