"""Defines the reinforcement learning `Engine`."""
import glob
import os
import queue
import random
import shutil
import sys
import time
import traceback
import typing
import warnings
from typing import Optional, Any, Dict, Union, List, Tuple
import logging
import json

import torch
import torch.distributions
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
from setproctitle import setproctitle as ptitle
from tensorboardX import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from onpolicy_sync.storage import RolloutStorage
from onpolicy_sync.vector_sampled_tasks import VectorSampledTasks
from rl_base.common import Loss
from rl_base.experiment_config import ExperimentConfig
from utils.experiment_utils import (
    ScalarMeanTracker,
    LinearDecay,
    set_deterministic_cudnn,
    set_seed,
    Builder,
    TrainingPipeline,
    PipelineStage,
)
from utils.tensor_utils import batch_observations

logger = logging.getLogger("embodiedrl")


def validate(
    config: ExperimentConfig,
    output_dir: str,
    read_from_parent: mp.Queue,
    write_to_parent: mp.Queue,
    seed: Optional[int] = None,
    deterministic_cudnn: bool = False,
):
    ptitle("Validation")
    evaluator = Validator(
        config=config,
        output_dir=output_dir,
        seed=seed,
        deterministic_cudnn=deterministic_cudnn,
    )
    evaluator.process_checkpoints(read_from_parent, write_to_parent)


class Engine(object):
    """The reinforcement learning primary controller.

    This `Engine` class handles all training, validation, and testing as
    well as logging and checkpointing. You are not expected to
    instantiate this class yourself, instead you should define an
    experiment which will then be used to instantiate an `Engine` and
    perform any desired tasks.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str,
        loaded_config_src_files: Optional[Dict[str, Tuple[str, str]]],
        seed: Optional[int] = None,
        mode: str = "train",
        deterministic_cudnn: bool = False,
    ):
        """Initializer.

        config : The ExperimentConfig defining the experiment to run.
        output_dir : Root directory at which checkpoints and logs should be saved.
        loaded_config_src_files : Paths to source config files used to create the experiment.
        seed : Seed used to encourage deterministic behavior (it is difficult to ensure
            completely deterministic behavior due to CUDA issues and nondeterminism
            in environments).
        mode : "train", "valid", or "test".
        deterministic_cudnn : Whether or not to use deterministic cudnn. If `True` this may lower
            training performance this is necessary (but not sufficient) if you desire
            deterministic behavior.
        """
        self.deterministic_cudnn = deterministic_cudnn
        self.seed = seed
        self.mode = mode.lower()
        assert self.mode in [
            "train",
            "valid",
            "test",
        ], "Only train, valid, test modes supported"

        self.training_pipeline: TrainingPipeline = config.training_pipeline()
        self.machine_params = config.machine_params(self.mode)

        self.device = "cpu"
        if len(self.machine_params["gpu_ids"]) > 0:
            if not torch.cuda.is_available():
                print(
                    "Warning: no CUDA devices available for gpu ids {}".format(
                        self.machine_params["gpu_ids"]
                    )
                )
            else:
                self.device = "cuda:%d" % self.machine_params["gpu_ids"][0]
                torch.cuda.set_device(self.device)  # type: ignore

        if self.deterministic_cudnn:
            set_deterministic_cudnn()

        seeds: Optional[List[int]] = None
        if self.seed is not None:
            set_seed(self.seed)
            seeds = self.worker_seeds(self.machine_params["nprocesses"])

        self.observation_set = None
        # if "observation_set" in self.machine_params:
        #     self.observation_set = self.machine_params["observation_set"].to(self.device)
        #     self.actor_critic = config.create_model(
        #         observation_set=self.observation_set
        #     ).to(self.device)
        # else:
        self.actor_critic = config.create_model().to(self.device)

        self.optimizer: Optional[  # type: ignore
            Union[optim.Optimizer, Builder[optim.Optimizer]]
        ] = None
        self.scheduler: Optional[
            Union[
                optim.lr_scheduler._LRScheduler,
                Builder[optim.lr_scheduler._LRScheduler],
            ]
        ] = None
        if mode == "train":
            self.optimizer = self.training_pipeline.optimizer
            if isinstance(self.optimizer, Builder):
                self.optimizer = typing.cast(Builder, self.optimizer)(
                    params=[
                        p for p in self.actor_critic.parameters() if p.requires_grad
                    ]
                )
            self.scheduler = self.training_pipeline.scheduler
            if isinstance(self.scheduler, Builder):
                self.scheduler = typing.cast(Builder, self.scheduler)(
                    optimizer=self.optimizer
                )

        self.vector_tasks = VectorSampledTasks(
            make_sampler_fn=config.make_sampler_fn,
            sampler_fn_args=self.get_sampler_fn_args(config, seeds),
        )

        self.output_dir = output_dir
        self.models_folder: Optional[str] = None

        self.configs_folder = os.path.join(output_dir, "used_configs")
        os.makedirs(self.configs_folder, exist_ok=True)
        if mode == "train":
            for file in loaded_config_src_files:
                base, module = loaded_config_src_files[file]
                parts = module.split(".")
                src_file = os.path.sep.join([base] + parts) + ".py"
                dst_file = (
                    os.path.join(self.configs_folder, os.path.join(*parts[1:])) + ".py"
                )
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy(src_file, dst_file)

        self.log_writer = None

        self.scalars = ScalarMeanTracker()

        self.total_updates = 0
        self.pipeline_stage = 0
        self.rollout_count = 0
        self.backprop_count = 0
        self.step_count = 0
        self.total_steps = 0
        self.last_log = 0
        self.last_save = 0

        # Fields defined when running setup_stage.
        # TODO: Lets encapsulate these better, perhaps in named
        #   tuple like data structure with sensible defaults.
        self.losses: Optional[Dict[str, Loss]] = None
        self.loss_weights: Optional[Dict[str, float]] = None
        self.stage_task_steps: Optional[int] = None
        self.steps_in_rollout: Optional[int] = None
        self.update_epochs: Optional[int] = None
        self.update_mini_batches: Optional[int] = None
        self.num_rollouts: Optional[int] = None
        self.gamma: Optional[float] = None
        self.use_gae: Optional[bool] = None
        self.gae_lambda: Optional[float] = None
        self.max_grad_norm: Optional[float] = None
        self.teacher_forcing: Optional[LinearDecay] = None
        self.local_start_time_str: Optional[str] = None
        self.deterministic_agent: Optional[bool] = None
        self.eval_process: Optional[mp.Process] = None
        self.last_scheduler_steps: Optional[int] = None

        self.experiment_name = config.tag()

        self.save_interval = self.training_pipeline.save_interval
        self.log_interval = self.training_pipeline.log_interval
        self.num_processes = self.machine_params["nprocesses"]

        self.config = config

        self.write_to_eval = None
        if self.mode == "train":
            self.mp_ctx = self.vector_tasks.mp_ctx
            if self.config.machine_params("valid")["nprocesses"] <= 0:
                print(
                    "No processes allocated to validation, no validation will be run."
                )
            else:
                self.write_to_eval = self.mp_ctx.Queue()
                self.eval_process = self.mp_ctx.Process(
                    target=validate,
                    args=(
                        self.config,
                        self.output_dir,
                        self.write_to_eval,
                        self.vector_tasks.metrics_out_queue,
                        self.seed,
                        self.deterministic_cudnn,
                    ),
                )
                self.eval_process.start()

        self._is_closed: bool = False

    @staticmethod
    def worker_seeds(nprocesses: int) -> List[int]:
        """Create a collection of seeds for workers."""
        return [random.randint(0, 2 ** (31) - 1) for _ in range(nprocesses)]

    def get_sampler_fn_args(
        self, config: ExperimentConfig, seeds: Optional[List[int]] = None
    ):
        devices = (
            self.machine_params["sampler_devices"]
            if "sampler_devices" in self.machine_params
            else self.machine_params["gpu_ids"]
        )

        if self.mode == "train":
            fn = config.train_task_sampler_args
        elif self.mode == "valid":
            fn = config.valid_task_sampler_args
        elif self.mode == "test":
            fn = config.test_task_sampler_args
        else:
            raise NotImplementedError(
                "self.mode must be one of `train`, `valid` or `test`."
            )

        return [
            fn(
                process_ind=it,
                total_processes=self.machine_params["nprocesses"],
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=self.deterministic_cudnn,
            )
            for it in range(self.machine_params["nprocesses"])
        ]

    def checkpoint_save(self) -> str:
        self.models_folder = os.path.join(
            self.output_dir, "checkpoints", self.local_start_time_str
        )
        os.makedirs(self.models_folder, exist_ok=True)

        if self.seed is not None:
            self.seed = self.worker_seeds(1)[0]
            set_seed(self.seed)

            seeds = self.worker_seeds(self.num_processes)
            self.vector_tasks.set_seeds(seeds)

        model_path = os.path.join(
            self.models_folder,
            "exp_{}__time_{}__stage_{:02d}__steps_{:012d}__seed_{}.pt".format(
                self.experiment_name,
                self.local_start_time_str,
                self.pipeline_stage,
                self.total_steps + self.step_count,
                self.seed,
            ),
        )

        save_dict = {
            "total_updates": self.total_updates,
            "total_steps": self.total_steps,
            "pipeline_stage": self.pipeline_stage,
            "rollout_count": self.rollout_count,
            "backprop_count": self.backprop_count,
            "step_count": self.step_count,
            "local_start_time_str": self.local_start_time_str,
            "optimizer_state_dict": self.optimizer.state_dict(),  # type: ignore
            "model_state_dict": self.actor_critic.state_dict(),
            "trainer_seed": self.seed,
        }

        if self.seed is not None:
            save_dict["worker_seeds"] = seeds

        if self.scheduler is not None:
            save_dict["scheduler_state"] = typing.cast(
                _LRScheduler, self.scheduler
            ).state_dict()
            save_dict["scheduler_steps"] = self.last_scheduler_steps

        torch.save(save_dict, model_path)
        return model_path

    def checkpoint_load(self, ckpt: Union[str, Dict[str, Any]], verbose=False) -> None:
        if isinstance(ckpt, str):
            if verbose:
                print("Loading checkpoint from %s" % ckpt)
            # Map location CPU is almost always better than mapping to a CUDA device.
            ckpt = torch.load(ckpt, map_location="cpu")

        ckpt = typing.cast(
            Dict[
                str, Union[Dict[str, Any], torch.Tensor, float, int, str, typing.List]
            ],
            ckpt,
        )

        self.actor_critic.load_state_dict(ckpt["model_state_dict"])
        self.step_count = ckpt["step_count"]  # type: ignore
        self.total_steps = ckpt["total_steps"]  # type: ignore

        if self.mode == "train":
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # type: ignore
            self.backprop_count = ckpt["backprop_count"]  # type: ignore
            self.rollout_count = ckpt["rollout_count"]  # type: ignore
            self.pipeline_stage = ckpt["pipeline_stage"]  # type: ignore
            self.total_updates = ckpt["total_updates"]  # type: ignore
            self.local_start_time_str = typing.cast(str, ckpt["local_start_time_str"])
            self.seed = typing.cast(int, ckpt["trainer_seed"])
            if self.seed is not None:
                set_seed(self.seed)
                seeds = self.worker_seeds(self.num_processes)
                assert (
                    seeds == ckpt["worker_seeds"]
                ), "worker seeds not matching stored seeds"
                self.vector_tasks.set_seeds(seeds)
            if self.scheduler is not None:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])  # type: ignore
                self.last_scheduler_steps = typing.cast(int, ckpt["scheduler_steps"])

    def process_eval_metrics(self, count=-1):
        unused = []
        while (not self.vector_tasks.metrics_out_queue.empty()) or (count > 0):
            try:
                if count < 0:
                    metric = self.vector_tasks.metrics_out_queue.get_nowait()
                else:
                    metric = self.vector_tasks.metrics_out_queue.get(timeout=1)
                if (
                    isinstance(metric, tuple) and metric[0] == "test_metrics"
                ):  # queue reused for test
                    unused.append(metric)
                else:
                    self.scalars.add_scalars(metric)
                    count -= 1
            except queue.Empty:
                pass

        for item in unused:
            self.vector_tasks.metrics_out_queue.put(item)

        return self.scalars.pop_and_reset()

    def log(self, count=-1):
        while (not self.vector_tasks.metrics_out_queue.empty()) or (count > 0):
            try:
                eval_metrics = {}
                if count < 0:
                    metric = self.vector_tasks.metrics_out_queue.get_nowait()
                else:
                    metric = self.vector_tasks.metrics_out_queue.get(timeout=1)
                    count -= 1
                if isinstance(metric, tuple):
                    pkg_type, info = metric
                    if pkg_type == "valid_metrics":
                        eval_metrics["valid"] = {k: v for k, v in info.items()}
                    elif pkg_type == "test_metrics":
                        eval_metrics["test"] = {k: v for k, v in info.items()}
                    else:
                        cscalars: Optional[Dict[str, Union[float, int]]] = None
                        if pkg_type == "update_package":
                            cscalars = {
                                "total_loss": info["total_loss"],
                            }
                            if "lr" in info:
                                cscalars["lr"] = info["lr"]
                            for loss in info["losses"]:
                                lossname = loss[:-5] if loss.endswith("_loss") else loss
                                for scalar in info["losses"][loss]:
                                    cscalars["/".join([lossname, scalar])] = info[
                                        "losses"
                                    ][loss][scalar]
                        elif pkg_type == "teacher_package":
                            cscalars = {k: v for k, v in info.items()}
                        else:
                            print("WARNING: Unknown info package {}".format(info))

                        if cscalars is not None:
                            self.scalars.add_scalars(cscalars)
                else:
                    self.scalars.add_scalars(metric)

                for mode in eval_metrics:
                    message = ["{}".format(mode)]
                    add_step = True
                    for k in eval_metrics[mode]:
                        if add_step:
                            message += ["{} steps:".format(eval_metrics[mode][k][1])]
                            add_step = False
                        self.log_writer.add_scalar(
                            "{}/".format(mode) + k,
                            eval_metrics[mode][k][0],
                            eval_metrics[mode][k][1],
                        )
                        message += [k + " {}".format(eval_metrics[mode][k][0])]
                    if len(eval_metrics[mode]) > 0:
                        logger.info(" ".join(message))

            except queue.Empty:
                pass

        tracked_means = self.scalars.pop_and_reset()
        message = ["train {} steps:".format(self.total_steps + self.step_count)]
        for k in tracked_means:
            self.log_writer.add_scalar(
                "train/" + k, tracked_means[k], self.total_steps + self.step_count,
            )
            message += [k + " {}".format(tracked_means[k])]
        if len(tracked_means) > 0:
            logger.info(" ".join(message))

    def update(self, rollouts) -> None:
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        for e in range(self.update_epochs):
            data_generator = rollouts.recurrent_generator(
                advantages, self.update_mini_batches
            )

            for bit, batch in enumerate(data_generator):
                actor_critic_output, hidden_states = self.actor_critic(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["masks"],
                )

                info: Dict[str, Any] = dict(
                    total_updates=self.total_updates,
                    backprop_count=self.backprop_count,
                    rollout_count=self.rollout_count,
                    epoch=e,
                    batch=bit,
                    losses={},
                )

                if self.scheduler is not None:
                    info["lr"] = self.optimizer.param_groups[0]["lr"]  # type: ignore

                self.optimizer.zero_grad()  # type: ignore
                total_loss: Optional[torch.FloatTensor] = None
                for loss_name in self.losses:
                    loss, loss_weight = (
                        self.losses[loss_name],
                        self.loss_weights[loss_name],
                    )

                    current_loss, current_info = loss.loss(batch, actor_critic_output)
                    if total_loss is None:
                        total_loss = loss_weight * current_loss
                    else:
                        total_loss = total_loss + loss_weight * current_loss

                    info["losses"][loss_name] = current_info
                assert total_loss is not None, "No losses specified?"

                if isinstance(total_loss, torch.Tensor):
                    info["total_loss"] = total_loss.item()
                    self.vector_tasks.metrics_out_queue.put(("update_package", info))

                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm, norm_type="inf"  # type: ignore
                    )
                    self.optimizer.step()  # type: ignore
                    self.backprop_count += 1
                else:
                    warnings.warn(
                        "Total loss ({}) was not a FloatTensor, it is a {}.".format(
                            total_loss, type(total_loss)
                        )
                    )

    def _preprocess_observations(self, batched_observations):
        if self.observation_set is None:
            return batched_observations
        return self.observation_set.get_observations(batched_observations)

    def apply_teacher_forcing(self, actions, step_observation):
        tf_mask_shape = step_observation["expert_action"].shape[:-1] + (1,)
        expert_actions = (
            step_observation["expert_action"].view(-1, 2)[:, 0].view(*tf_mask_shape)
        )
        expert_action_exists_mask = (
            step_observation["expert_action"].view(-1, 2)[:, 1].view(*tf_mask_shape)
        )
        teacher_forcing_mask = (
            torch.distributions.bernoulli.Bernoulli(
                torch.tensor(self.teacher_forcing(self.step_count))
            )
            .sample(tf_mask_shape)
            .long()
            .to(self.device)
        ) * expert_action_exists_mask
        actions = (
            teacher_forcing_mask * expert_actions + (1 - teacher_forcing_mask) * actions
        )

        return (
            actions,
            {"teacher_forcing_mask": teacher_forcing_mask},
        )

    def collect_rollout_step(self, rollouts):
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            actor_critic_output, recurrent_hidden_states = self.actor_critic(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        actions = (
            actor_critic_output.distributions.sample()
            if not self.deterministic_agent
            else actor_critic_output.distributions.mode()
        )

        if (
            self.teacher_forcing is not None
            and self.teacher_forcing(self.step_count) > 0
        ):
            actions, enforce_info = self.apply_teacher_forcing(
                actions, step_observation
            )
            teacher_force_info = {
                "teacher_ratio": enforce_info["teacher_forcing_mask"].sum().item()
                / actions.nelement(),
                "teacher_enforcing": self.teacher_forcing(self.step_count),
            }
            self.vector_tasks.metrics_out_queue.put(
                ("teacher_package", teacher_force_info)
            )

        if self.mode == "train":
            self.step_count += actions.nelement()

        outputs = self.vector_tasks.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        rewards = rewards.unsqueeze(1)

        # If done then clean the history of observations.
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float32,
            device=self.device,
        )

        npaused, keep, batch = self.remove_paused(observations)

        rollouts.reshape(keep)

        rollouts.insert(
            self._preprocess_observations(batch),
            recurrent_hidden_states[:, keep],
            actions[keep],
            actor_critic_output.distributions.log_probs(actions)[keep],
            actor_critic_output.values[keep],
            rewards[keep],
            masks[keep],
        )

        return npaused

    def remove_paused(self, observations):
        paused, keep, running = [], [], []
        for it, obs in enumerate(observations):
            if obs is None:
                paused.append(it)
            else:
                keep.append(it)
                running.append(obs)

        for p in reversed(paused):
            self.vector_tasks.pause_at(p)

        batch = batch_observations(running, device=self.device)

        return len(paused), keep, batch

    def initialize_rollouts(self, rollouts):
        observations = self.vector_tasks.get_observations()
        npaused, keep, batch = self.remove_paused(observations)
        rollouts.reshape(keep)
        rollouts.to(self.device)
        rollouts.insert_initial_observations(self._preprocess_observations(batch))
        return npaused

    def train(self, rollouts):
        try:
            self.initialize_rollouts(rollouts)

            while self.rollout_count < self.num_rollouts:
                for step in range(self.steps_in_rollout):
                    self.collect_rollout_step(rollouts)

                with torch.no_grad():
                    step_observation = {
                        k: v[-1] for k, v in rollouts.observations.items()
                    }

                    actor_critic_output, _ = self.actor_critic(
                        step_observation,
                        rollouts.recurrent_hidden_states[-1],
                        rollouts.prev_actions[-1],
                        rollouts.masks[-1],
                    )

                rollouts.compute_returns(
                    actor_critic_output.values,
                    self.use_gae,
                    self.gamma,
                    self.gae_lambda,
                )

                self.update(rollouts)

                rollouts.after_update()

                if self.scheduler is not None:
                    new_scheduler_steps = self.total_steps + self.step_count
                    for step in range(
                        self.last_scheduler_steps + 1, new_scheduler_steps + 1
                    ):
                        self.scheduler.step(step)
                    self.last_scheduler_steps = new_scheduler_steps

                if (
                    self.step_count - self.last_log >= self.log_interval
                    or self.rollout_count == self.num_rollouts
                ):
                    self.log()
                    self.last_log = self.step_count

                self.rollout_count += 1

                # save for every interval-th episode or for the last epoch
                if (
                    self.save_interval > 0
                    and (
                        self.step_count - self.last_save >= self.save_interval
                        or self.rollout_count == self.num_rollouts
                    )
                ) and self.models_folder != "":
                    model_path = self.checkpoint_save()
                    if self.write_to_eval is not None:
                        self.write_to_eval.put(("eval", model_path))
                    self.last_save = self.step_count
        finally:
            self.close()

    def setup_stage(
        self,
        losses: Dict[str, Loss],
        loss_weights: Dict[str, float],
        steps_in_rollout: int,
        stage_task_steps: int,
        update_epochs: int,
        update_mini_batches: int,
        gamma: float,
        use_gae: bool,
        gae_lambda: float,
        max_grad_norm: float,
        teacher_forcing: Optional[LinearDecay] = None,
        deterministic_agent: bool = False,
    ):
        self.losses = losses
        self.loss_weights = loss_weights

        self.stage_task_steps = stage_task_steps
        self.steps_in_rollout = steps_in_rollout
        self.update_epochs = update_epochs
        self.update_mini_batches = update_mini_batches

        self.num_rollouts = (
            int(self.stage_task_steps) // self.steps_in_rollout
        ) // self.num_processes
        message = "Using %d rollouts, %d steps (from %d)" % (
            self.num_rollouts,
            self.num_rollouts * self.num_processes * self.steps_in_rollout,
            self.stage_task_steps,
        )
        logger.info(message)

        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.max_grad_norm = max_grad_norm

        self.teacher_forcing = teacher_forcing

        self.deterministic_agent = deterministic_agent

    def _get_loss(self, loss_name):
        assert (
            loss_name in self.training_pipeline.named_losses
        ), "undefined referenced loss"
        if isinstance(self.training_pipeline.named_losses[loss_name], Builder):
            return self.training_pipeline.named_losses[loss_name]()
        else:
            return self.training_pipeline.named_losses[loss_name]

    def _load_losses(self, stage: PipelineStage):
        stage_losses = dict()
        for loss_name in stage.loss_names:
            stage_losses[loss_name] = self._get_loss(loss_name)

        loss_weights_list = (
            stage.loss_weights
            if stage.loss_weights is not None
            else [1.0] * len(stage.loss_names)
        )
        stage_loss_weights = {
            name: weight for name, weight in zip(stage.loss_names, loss_weights_list)
        }

        return stage_losses, stage_loss_weights

    def _stage_value(self, stage, field):
        if hasattr(stage, field) and getattr(stage, field) is not None:
            return getattr(stage, field)

        if (
            hasattr(self.training_pipeline, field)
            and getattr(self.training_pipeline, field) is not None
        ):
            return getattr(self.training_pipeline, field)

        if field in self.machine_params:
            return self.machine_params[field]

        raise RuntimeError("missing value for {}".format(field))

    @property
    def log_writer_path(self) -> str:
        return os.path.join(
            self.output_dir, "tb", self.experiment_name, self.local_start_time_str
        )

    @property
    def metric_path(self) -> str:
        return os.path.join(
            self.output_dir, "metrics", self.experiment_name, self.local_start_time_str
        )

    def get_checkpoint_path(self, checkpoint_file_name: str) -> str:
        checkpoint_start_time = [
            s for s in checkpoint_file_name.split("__") if "time_" in s
        ][0].replace("time_", "")

        expected_path = os.path.join(
            self.output_dir, "checkpoints", checkpoint_start_time, checkpoint_file_name
        )
        if os.path.exists(expected_path):
            return expected_path
        else:
            print(
                (
                    "Could not find checkpoint with file name {}\n"
                    "under expected path {}.\n"
                    "Attempting to find the checkpoint elsewhere under the working directory.\n"
                ).format(checkpoint_file_name, expected_path)
            )

            ckpts = glob.glob("./**/{}".format(checkpoint_file_name), recursive=True)

            if len(ckpts) == 0:
                raise RuntimeError(
                    "Could not find {} anywhere"
                    " the working directory.".format(checkpoint_file_name)
                )
            elif len(ckpts) > 1:
                raise RuntimeError("Found too many checkpoint paths {}.".format(ckpts))
            else:
                return ckpts[0]

    def run_pipeline(self, checkpoint_file_name: Optional[str] = None):
        start_time = time.time()
        self.local_start_time_str = time.strftime(
            "%Y-%m-%d_%H-%M-%S", time.localtime(start_time)
        )
        self.log_writer = SummaryWriter(log_dir=self.log_writer_path)

        if self.scheduler is not None:
            self.last_scheduler_steps = 0

        if checkpoint_file_name is not None:
            self.checkpoint_load(
                self.get_checkpoint_path(checkpoint_file_name), verbose=True
            )

        for stage_num, stage in self.training_pipeline.iterator_starting_at(
            self.pipeline_stage
        ):
            assert stage_num == self.pipeline_stage

            self.last_log = self.step_count - self.log_interval
            self.last_save = self.step_count

            stage_losses, stage_weights = self._load_losses(stage)

            self.setup_stage(
                losses=stage_losses,
                loss_weights=stage_weights,
                steps_in_rollout=self._stage_value(stage, "num_steps"),
                stage_task_steps=self._stage_value(stage, "end_criterion"),
                update_epochs=self._stage_value(stage, "update_repeats"),
                update_mini_batches=self._stage_value(stage, "num_mini_batch"),
                gamma=self._stage_value(stage, "gamma"),
                use_gae=self._stage_value(stage, "use_gae"),
                gae_lambda=self._stage_value(stage, "gae_lambda"),
                max_grad_norm=self._stage_value(stage, "max_grad_norm"),
                teacher_forcing=stage.teacher_forcing,
            )

            self.train(
                RolloutStorage(
                    self.steps_in_rollout,
                    self.num_processes,
                    self.actor_critic.action_space,
                    self.actor_critic.recurrent_hidden_state_size,
                    num_recurrent_layers=self.actor_critic.num_recurrent_layers,
                )
            )

            self.total_updates += self.num_rollouts
            self.pipeline_stage += 1

            self.rollout_count = 0
            self.backprop_count = 0
            self.total_steps += self.step_count
            self.step_count = 0

        logger.info("\n\nTRAINING COMPLETE.\n\n")
        self.close()

    def process_checkpoints(
        self,
        read_from_parent: mp.Queue,
        write_to_parent: mp.Queue,
        deterministic_agent: bool = True,
    ):
        assert (
            self.mode != "train"
        ), "process_checkpoints only to be called from a valid or test instance"
        self.deterministic_agent = deterministic_agent
        self.teacher_forcing = None

        try:
            new_data = False
            command: Optional[str] = None
            data: Any = None
            while True:
                while (not new_data) or (not read_from_parent.empty()):
                    try:
                        command, data = read_from_parent.get_nowait()
                        new_data = True
                    except queue.Empty:
                        time.sleep(1)
                        pass

                if command == "eval":
                    scalars = self.run_eval(checkpoint_file_name=data)
                    write_to_parent.put(("valid_metrics", scalars))
                elif command in ["quit", "exit", "close"]:
                    self.close(verbose=False)
                    sys.exit()
                else:
                    raise NotImplementedError()

                new_data = False
        except KeyboardInterrupt:
            print("Eval KeyboardInterrupt")

    def run_eval(self, checkpoint_file_name: str, rollout_steps=1):
        self.checkpoint_load(checkpoint_file_name, verbose=False)

        rollouts = RolloutStorage(
            rollout_steps,
            self.num_processes,
            self.actor_critic.action_space,
            self.actor_critic.recurrent_hidden_state_size,
            num_recurrent_layers=self.actor_critic.num_recurrent_layers,
        )

        num_paused = self.initialize_rollouts(rollouts)
        steps = 0
        while num_paused < self.num_processes:
            num_paused += self.collect_rollout_step(rollouts)
            steps += 1
            if steps % rollout_steps == 0:
                rollouts.after_update()

        self.vector_tasks.resume_all()
        self.vector_tasks.reset_all()

        return {
            k: (v, self.total_steps + self.step_count)
            for k, v in self.process_eval_metrics(count=self.num_processes).items()
        }

    def get_checkpoint_files(
        self,
        experiment_date: str,
        checkpoint_file_name: Optional[str] = None,
        skip_checkpoints: int = 0,
    ):
        if checkpoint_file_name is not None:
            return [checkpoint_file_name]
        files = glob.glob(
            os.path.join(self.output_dir, "checkpoints", experiment_date, "exp_*.pt")
        )
        files = sorted(files)
        return files[:: skip_checkpoints + 1] + (
            [files[-1]]
            if skip_checkpoints > 0 and len(files) % (skip_checkpoints + 1) != 1
            else []
        )

    def step_from_checkpoint(self, name):
        parts = name.split("__")
        for part in parts:
            if "steps_" in part:
                return int(part.split("_")[-1])
        return -1

    def run_test(
        self,
        experiment_date: str,
        checkpoint_file_name: Optional[str] = None,
        skip_checkpoints=0,
        rollout_steps=1,
        deterministic_agent=True,
    ):
        assert (
            self.mode != "train"
        ), "run_test only to be called from a valid or test instance"
        self.deterministic_agent = deterministic_agent
        self.teacher_forcing = None

        test_start_time_str = time.strftime(
            "%Y-%m-%d_%H-%M-%S", time.localtime(time.time())
        )

        self.local_start_time_str = experiment_date

        checkpoints = self.get_checkpoint_files(
            experiment_date, checkpoint_file_name, skip_checkpoints
        )

        suffix = "__test_{}".format(test_start_time_str)
        self.log_writer = SummaryWriter(
            log_dir=self.log_writer_path, filename_suffix=suffix,
        )

        all_results = []
        for it, checkpoint_file_name in enumerate(checkpoints):
            step = self.step_from_checkpoint(checkpoint_file_name)
            logger.info("{}/{} {} steps".format(it + 1, len(checkpoints), step,))
            scalars = self.run_eval(checkpoint_file_name, rollout_steps)
            # logger.info("metrics", {k: v[0] for k, v in scalars.items()})
            self.vector_tasks.metrics_out_queue.put(("test_metrics", scalars))
            results = {scalar: scalars[scalar][0] for scalar in scalars}
            results.update({"training_steps": step})
            all_results.append(results)
            self.log(count=1)

        os.makedirs(self.metric_path, exist_ok=True)
        fname = os.path.join(self.metric_path, "metrics" + suffix + ".json")
        with open(fname, "w") as f:
            json.dump(all_results, f, indent=4)
        logger.info("Metrics saved in {}".format(fname))

    def close(self, verbose=True):
        if self._is_closed:
            return

        def printif(s: Union[str, Exception]):
            if verbose:
                if isinstance(s, str):
                    print(s)
                else:
                    traceback.print_exc()

        try:
            printif("Closing Engine.vector_tasks.")
            self.vector_tasks.close()
            printif("Closed.")
        except Exception as e:
            printif("Exception raised when closing Engine.vector_tasks:")
            printif(e)
            pass

        printif("\n\n")
        try:
            printif("Closing Engine.eval_process")
            eval: mp.Process = getattr(self, "eval_process", None)
            if eval is not None:
                self.write_to_eval.put(("exit", None))
                eval.join(5)
                self.eval_process = None
            printif("Closed.")
        except Exception as e:
            printif("Exception raised when closing Engine.vector_tasks:")
            printif(e)
            pass

        printif("\n\n")
        try:
            printif("Closing Engine.log_writer")
            log_writer = getattr(self, "log_writer", None)
            if log_writer is not None:
                log_writer.close()
                self.log_writer = None
            printif("Closed.")
        except Exception as e:
            printif("Exception raised when closing Engine.log_writer:")
            printif(e)
            pass

        self._is_closed = True

    def __del__(self):
        self.close(verbose=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(verbose=False)


class Trainer(Engine):
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str,
        loaded_config_src_files: Optional[Dict[str, Tuple[str, str]]],
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        *args,
        **argv
    ):
        super().__init__(
            config=config,
            loaded_config_src_files=loaded_config_src_files,
            output_dir=output_dir,
            seed=seed,
            mode="train",
            deterministic_cudnn=deterministic_cudnn,
        )


class Validator(Engine):
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        *args,
        **argv
    ):
        super().__init__(
            config=config,
            loaded_config_src_files=None,
            output_dir=output_dir,
            seed=seed,
            mode="valid",
            deterministic_cudnn=deterministic_cudnn,
        )


class Tester(Engine):
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        *args,
        **argv
    ):
        super().__init__(
            config=config,
            loaded_config_src_files=None,
            output_dir=output_dir,
            seed=seed,
            mode="test",
            deterministic_cudnn=deterministic_cudnn,
        )
