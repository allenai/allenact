import math
from typing import Optional, Sequence

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_zeroshot_rgb_clipgru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig as BaseConfig,
)
from allenact.utils.experiment_utils import (
    Builder,
    LinearDecay,
    MultiLinearDecay,
    TrainingPipeline,
    PipelineStage,
)


class DistributedObjectNavRoboThorRGBPPOExperimentConfig(BaseConfig):
    def tag(self) -> str:
        return "ObjectNavRoboThorRGBPPO_CLIP_zeroshot"

    THOR_COMMIT_ID = "91139c909576f3bf95a187c5b02c6fd455d06b48"

    THOR_IS_HEADLESS = True

    def env_args(self):
        res = super().env_args()
        res.pop("commit_id", None)
        return res

    # %%
    def __init__(
        self,
        distributed_nodes: int = 1,
        num_train_processes: Optional[int] = None,
        train_gpu_ids: Optional[Sequence[int]] = None,
        val_gpu_ids: Optional[Sequence[int]] = None,
        test_gpu_ids: Optional[Sequence[int]] = None,
    ):
        super().__init__(num_train_processes, train_gpu_ids, val_gpu_ids, test_gpu_ids)
        self.distributed_nodes = distributed_nodes

    # %%
    def machine_params(self, mode="train", **kwargs):
        params = super().machine_params(mode, **kwargs)

        if mode == "train":
            params.devices = params.devices * self.distributed_nodes
            params.nprocesses = params.nprocesses * self.distributed_nodes
            params.sampler_devices = params.sampler_devices * self.distributed_nodes

            if "machine_id" in kwargs:
                machine_id = kwargs["machine_id"]
                assert (
                    0 <= machine_id < self.distributed_nodes
                ), f"machine_id {machine_id} out of range [0, {self.distributed_nodes - 1}]"

                local_worker_ids = list(
                    range(
                        len(self.train_gpu_ids) * machine_id,
                        len(self.train_gpu_ids) * (machine_id + 1),
                    )
                )

                params.set_local_worker_ids(local_worker_ids)

            # Confirm we're setting up train params nicely:
            print(
                f"devices {params.devices}"
                f"\nnprocesses {params.nprocesses}"
                f"\nsampler_devices {params.sampler_devices}"
                f"\nlocal_worker_ids {params.local_worker_ids}"
            )
        elif mode == "valid":
            # Use all GPUs at their maximum capacity for training
            # (you may run validation in a separate machine)
            params.nprocesses = (0,)

        return params

    # %%
    @staticmethod
    def lr_scheduler(small_batch_steps, transition_steps, ppo_steps, lr_scaling):
        safe_small_batch_steps = int(small_batch_steps * 1.02)
        large_batch_and_lr_steps = ppo_steps - safe_small_batch_steps - transition_steps

        # Learning rate after small batch steps (assuming decay to 0)
        break1 = 1.0 - safe_small_batch_steps / ppo_steps

        # Initial learning rate for large batch (after transition from initial to large learning rate)
        break2 = lr_scaling * (
            1.0 - (safe_small_batch_steps + transition_steps) / ppo_steps
        )
        return MultiLinearDecay(
            [
                # Base learning rate phase for small batch (with linear decay towards 0)
                LinearDecay(steps=safe_small_batch_steps, startp=1.0, endp=break1,),
                # Allow the optimizer to adapt its statistics to the changes with a larger learning rate
                LinearDecay(steps=transition_steps, startp=break1, endp=break2,),
                # Scaled learning rate phase for large batch (with linear decay towards 0)
                LinearDecay(steps=large_batch_and_lr_steps, startp=break2, endp=0,),
            ]
        )

    # %%
    def training_pipeline(self, **kwargs):
        # These params are identical to the baseline configuration for 60 samplers (1 machine)
        ppo_steps = int(300e6)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 4
        num_steps = 128
        save_interval = 5000000
        log_interval = 10000 if torch.cuda.is_available() else 1
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5

        # We add 30 million steps for small batch learning
        small_batch_steps = int(30e6)
        # And a short transition phase towards large learning rate
        # (see comment in the `lr_scheduler` helper method
        transition_steps = int(2 / 3 * self.distributed_nodes * 1e6)

        # Find exact number of samplers per GPU
        assert (
            self.num_train_processes % len(self.train_gpu_ids) == 0
        ), "Expected uniform number of samplers per GPU"
        samplers_per_gpu = self.num_train_processes // len(self.train_gpu_ids)

        # Multiply num_mini_batch by the largest divisor of
        # samplers_per_gpu to keep all batches of same size:
        num_mini_batch_multiplier = [
            i
            for i in reversed(
                range(1, min(samplers_per_gpu // 2, self.distributed_nodes) + 1)
            )
            if samplers_per_gpu % i == 0
        ][0]

        # Multiply update_repeats so that the product of this factor and
        # num_mini_batch_multiplier is >= self.distributed_nodes:
        update_repeats_multiplier = int(
            math.ceil(self.distributed_nodes / num_mini_batch_multiplier)
        )

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"ppo_loss": PPO(**PPOConfig, show_ratios=False)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                # We increase the number of batches for the first stage to reach an
                # equivalent number of updates per collected rollout data as in the
                # 1 node/60 samplers setting
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=small_batch_steps,
                    num_mini_batch=num_mini_batch * num_mini_batch_multiplier,
                    update_repeats=update_repeats * update_repeats_multiplier,
                ),
                # The we proceed with the base configuration (leading to larger
                # batches due to the increased number of samplers)
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=ppo_steps - small_batch_steps,
                ),
            ],
            # We use the MultiLinearDecay curve defined by the helper function,
            # setting the learning rate scaling as the square root of the number
            # of nodes. Linear scaling might also works, but we leave that
            # check to the reader.
            lr_scheduler_builder=Builder(
                LambdaLR,
                {
                    "lr_lambda": self.lr_scheduler(
                        small_batch_steps=small_batch_steps,
                        transition_steps=transition_steps,
                        ppo_steps=ppo_steps,
                        lr_scaling=math.sqrt(self.distributed_nodes),
                    )
                },
            ),
        )
