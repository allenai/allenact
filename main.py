import os
from collections import deque
from typing import Optional

import torch

from configs.util import Builder
from onpolicy_sync.arguments import get_args
from onpolicy_sync.storage import RolloutStorage
from onpolicy_sync.vector_task import VectorSampledTasks
from rl_base.experiment_config import ExperimentConfig
from onpolicy_sync.trainer import Trainer


def run_pipeline(
    config: ExperimentConfig,
    output_dir: str,
    checkpoint_file_name: Optional[str] = None,
):
    train_pipeline = config.training_pipeline()

    actor_critic = config.create_model()  # TODO Don't we always have to create it?

    optimizer = train_pipeline["optimizer"]
    if isinstance(optimizer, Builder):  # TODO Should it always be true (?)
        optimizer = optimizer(
            params=[p for p in actor_critic.parameters() if p.requires_grad]
        )

    sampler_fn_args = [
        config.train_task_sampler_args(it, train_pipeline["nprocesses"])
        for it in range(train_pipeline["nprocesses"])
    ]

    vectask = VectorSampledTasks(
        make_sampler_fn=config.make_sampler_fn, sampler_fn_args=sampler_fn_args
    )

    nsteps = 0
    info = deque(maxlen=100)

    ckpt_dict = None
    if checkpoint_file_name is not None:
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = torch.load(checkpoint_file_name, map_location="cpu")
        print("Loaded checkpoint from %s" % checkpoint_file_name)

    for sit, stage in enumerate(train_pipeline["pipeline"]):
        stage_limit = stage["criterion"]
        stage_losses = dict()
        stage_weights = {name: 1.0 for name in stage["losses"]}
        for name in stage["losses"]:
            if isinstance(train_pipeline[name], Builder):
                stage_losses[name] = train_pipeline[name](optimizer=optimizer)
            else:
                stage_losses[name] = train_pipeline[name]

        trainer = Trainer(
            vector_tasks=vectask,
            actor_critic=actor_critic,
            losses=stage_losses,
            loss_weights=stage_weights,
            optimizer=optimizer,
            rollout_steps=train_pipeline["num_steps"],
            stage_task_steps=stage_limit,
            update_epochs=train_pipeline["update_repeats"],
            update_mini_batches=train_pipeline["num_mini_batch"],
            num_processes=train_pipeline["nprocesses"],
            gamma=0.99,
            use_gae=True,
            gae_lambda=1.0,
            max_grad_norm=0.5,
            tracker=info,
            models_folder=os.path.join(output_dir, "models"),
            save_interval=10000,
            pipeline_stage=sit,
            teacher_forcing=None,
        )

        if ckpt_dict is not None and sit == ckpt_dict["pipeline_stage"]:
            trainer.checkpoint_load(ckpt_dict)
            ckpt_dict = None

        if ckpt_dict is None:
            rollouts = RolloutStorage(
                train_pipeline["num_steps"],
                train_pipeline["nprocesses"],
                vectask.observation_space.shape,
                actor_critic.action_space,
                actor_critic.recurrent_hidden_state_size,
            )

            trainer.train(rollouts)

            rollouts = None

        nsteps += stage_limit  # TODO


def main():
    args = get_args()
    if args.experiment == "object_nav_thor":
        from experiments.object_nav_thor import ObjectNavThorExperimentConfig as config
    run_pipeline(config, args.output_dir, args.checkpoint)


if __name__ == "__main__":
    main()
