import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from core.algorithms.onpolicy_sync.losses.imitation import Imitation
from projects.objectnav_baselines.experiments.objectnav_base import ObjectNavBaseConfig
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay


class ObjectNavMixInDAggerConfig(ObjectNavBaseConfig):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    def __init__(self):
        super().__init__()
        self.REWARD_CONFIG["shaping_weight"] = 0

    def training_pipeline(self, **kwargs):
        training_steps = int(300000000)
        tf_steps = int(5e6)
        anneal_steps = int(5e6)
        il_no_tf_steps = training_steps - tf_steps - anneal_steps
        assert il_no_tf_steps > 0

        lr = 3e-4
        num_mini_batch = 2 if torch.cuda.is_available() else 1
        update_repeats = 4
        num_steps = 30
        save_interval = 5000000
        log_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"imitation_loss": Imitation(),},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=tf_steps,
                    teacher_forcing=LinearDecay(startp=1.0, endp=1.0, steps=tf_steps,),
                ),
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=anneal_steps + il_no_tf_steps,
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=0.0, steps=anneal_steps,
                    ),
                ),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=training_steps)},
            ),
        )
