import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import (
    Builder,
    PipelineStage,
    TrainingPipeline,
    LinearDecay,
)
from allenact.base_abstractions.sensor import ExpertActionSensor
from projects.tutorials.object_nav_ithor_ppo_one_object import (
    ObjectNavThorPPOExperimentConfig,
    ObjectNaviThorGridTask,
)


class ObjectNavThorDaggerThenPPOExperimentConfig(ObjectNavThorPPOExperimentConfig):
    """A simple object navigation experiment in THOR.

    Training with DAgger and then PPO.
    """

    SENSORS = ObjectNavThorPPOExperimentConfig.SENSORS + [
        ExpertActionSensor(
            action_space=len(ObjectNaviThorGridTask.class_action_names()),
        ),
    ]

    @classmethod
    def tag(cls):
        return "ObjectNavThorDaggerThenPPO"

    @classmethod
    def training_pipeline(cls, **kwargs):
        dagger_steos = int(1e4)
        ppo_steps = int(1e6)
        lr = 2.5e-4
        num_mini_batch = 2 if not torch.cuda.is_available() else 6
        update_repeats = 4
        num_steps = 128
        metric_accumulate_interval = cls.MAX_STEPS * 10  # Log every 10 max length tasks
        save_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.5

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=metric_accumulate_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={
                "ppo_loss": PPO(clip_decay=LinearDecay(ppo_steps), **PPOConfig),
                "imitation_loss": Imitation(),  # We add an imitation loss.
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    teacher_forcing=LinearDecay(
                        startp=1.0,
                        endp=0.0,
                        steps=dagger_steos,
                    ),
                    max_stage_steps=dagger_steos,
                ),
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=ppo_steps,
                ),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )
