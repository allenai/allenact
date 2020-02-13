"""Defines the reinforcement learning `OnPolicyRLEngine`."""
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
from multiprocessing.context import BaseContext
from typing import Optional, Any, Dict, Union, List, Tuple
import logging
import json
from collections import OrderedDict

import torch
import torch.distributions
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
from setproctitle import setproctitle as ptitle
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

from onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from onpolicy_sync.storage import RolloutStorage
from onpolicy_sync.vector_sampled_tasks import VectorSampledTasks
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
from utils.tensor_utils import batch_observations, SummaryWriter, tensor_to_video

LOGGER = logging.getLogger("embodiedrl")


def validate(
    config: ExperimentConfig,
    output_dir: str,
    read_from_parent: mp.Queue,
    write_to_parent: mp.Queue,
    seed: Optional[int] = None,
    deterministic_cudnn: bool = False,
    mp_ctx: Optional[BaseContext] = None,
):
    ptitle("Validation")
    evaluator = OnPolicyValidator(
        config=config,
        output_dir=output_dir,
        seed=seed,
        deterministic_cudnn=deterministic_cudnn,
        mp_ctx=mp_ctx,
    )
    evaluator.process_checkpoints(read_from_parent, write_to_parent)


class OnPolicyRLEngine(object):
    """The reinforcement learning primary controller.

    This `OnPolicyRLEngine` class handles all training, validation, and testing as
    well as logging and checkpointing. You are not expected to
    instantiate this class yourself, instead you should define an
    experiment which will then be used to instantiate an `OnPolicyRLEngine` and
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
        mp_ctx: Optional[BaseContext] = None,
        extra_tag: str = "",
    ):
        """Initializer.

        # Parameters

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
        extra_tag : An additional label to add to the experiment when saving tensorboard logs.
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
                LOGGER.warning(
                    "Warning: no CUDA devices available for gpu ids {}".format(
                        self.machine_params["gpu_ids"]
                    )
                )
            else:
                self.device = "cuda:%d" % self.machine_params["gpu_ids"][0]
                torch.cuda.set_device(self.device)  # type: ignore

        if self.deterministic_cudnn:
            set_deterministic_cudnn()

        if self.seed is not None:
            set_seed(self.seed)

        self.observation_set = None
        if "observation_set" in self.machine_params:
            self.observation_set = self.machine_params["observation_set"].to(
                self.device
            )
            self.actor_critic = config.create_model(
                observation_set=self.observation_set
            ).to(self.device)
        else:
            self.actor_critic = config.create_model().to(self.device)

        self.optimizer: Optional[optim.Optimizer] = None  # type: ignore
        self.lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        if mode == "train":
            self.optimizer = self.training_pipeline.optimizer_builder(
                params=[p for p in self.actor_critic.parameters() if p.requires_grad]
            )

            if self.training_pipeline.lr_scheduler_builder is not None:
                self.lr_scheduler = self.training_pipeline.lr_scheduler_builder(
                    optimizer=self.optimizer
                )

        self._vector_tasks: Optional[VectorSampledTasks] = None

        self.output_dir = output_dir
        self.models_folder: Optional[str] = None

        self.configs_folder = os.path.join(output_dir, "used_configs")
        os.makedirs(self.configs_folder, exist_ok=True)

        self.loaded_config_src_files = loaded_config_src_files

        self.log_writer: Optional[SummaryWriter] = None

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
        self.losses: Optional[Dict[str, AbstractActorCriticLoss]] = None
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
        self.advance_scene_rollout_period: Optional[int] = None
        self.teacher_forcing: Optional[LinearDecay] = None
        self.local_start_time_str: Optional[str] = None
        self.deterministic_agent: Optional[bool] = None
        self.eval_process: Optional[mp.Process] = None

        self.experiment_name = config.tag()

        self.save_interval = self.training_pipeline.save_interval
        self.log_interval = self.training_pipeline.log_interval
        self.num_processes = self.machine_params["nprocesses"]

        self.config = config
        self.extra_tag = extra_tag

        self.write_to_eval = None
        self.mp_ctx: Optional[BaseContext] = mp_ctx
        if self.mode == "train":
            self.mp_ctx = self.vector_tasks.mp_ctx
            if self.config.machine_params("valid")["nprocesses"] <= 0:
                print(
                    "No processes allocated to validation, no validation will be run."
                )
            else:
                self.write_to_eval = self.mp_ctx.Queue()
                self.eval_process = self.mp_ctx.Process(  # type: ignore
                    target=validate,
                    args=(
                        self.config,
                        self.output_dir,
                        self.write_to_eval,
                        self.vector_tasks.metrics_out_queue,
                        self.seed,
                        self.deterministic_cudnn,
                        self.mp_ctx,
                    ),
                )
                self.eval_process.start()

        self._is_closed: bool = False

    @property
    def vector_tasks(self):
        if self._vector_tasks is None:
            seeds = self.worker_seeds(
                self.machine_params["nprocesses"], initial_seed=self.seed
            )
            self._vector_tasks = VectorSampledTasks(
                make_sampler_fn=self.config.make_sampler_fn,
                sampler_fn_args=self.get_sampler_fn_args(self.config, seeds),
                multiprocessing_start_method="forkserver"
                if self.mp_ctx is None
                else None,
                mp_ctx=self.mp_ctx,
            )
        return self._vector_tasks

    @staticmethod
    def worker_seeds(nprocesses: int, initial_seed: Optional[int]) -> List[int]:
        """Create a collection of seeds for workers."""
        if initial_seed is not None:
            rstate = random.getstate()
            random.seed(initial_seed)
        seeds = [random.randint(0, 2 ** (31) - 1) for _ in range(nprocesses)]
        if initial_seed is not None:
            random.setstate(rstate)
        return seeds

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
            self.seed = self.worker_seeds(1, None)[0]
            set_seed(self.seed)

            seeds = self.worker_seeds(self.num_processes, None)
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
            "extra_tag": self.extra_tag,
        }

        if self.seed is not None:
            save_dict["worker_seeds"] = seeds

        if self.lr_scheduler is not None:
            save_dict["scheduler_state"] = typing.cast(
                _LRScheduler, self.lr_scheduler
            ).state_dict()

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
                seeds = self.worker_seeds(self.num_processes, None)
                assert (
                    seeds == ckpt["worker_seeds"]
                ), "worker seeds not matching stored seeds"
                self.vector_tasks.set_seeds(seeds)
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(ckpt["scheduler_state"])  # type: ignore

            self.extra_tag = typing.cast(
                str,
                ckpt["extra_tag"]
                if (
                    "extra_tag" in ckpt
                    and ckpt["extra_tag"] != ""
                    and self.extra_tag == ""
                )
                else self.extra_tag,
            )

    def process_eval_metrics(self, count=-1):
        unused = []
        used = []
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
                    self.scalars.add_scalars(
                        {k: v for k, v in metric.items() if k != "task_info"}
                    )
                    used.append(metric)
                    count -= 1
            except queue.Empty:
                pass

        for item in unused:
            self.vector_tasks.metrics_out_queue.put(item)

        return self.scalars.pop_and_reset(), used

    def log(self, count=-1):
        train_metrics = []
        losses = []
        teachers = []
        while (not self.vector_tasks.metrics_out_queue.empty()) or (count > 0):
            try:
                if count < 0:
                    metric = self.vector_tasks.metrics_out_queue.get_nowait()
                else:
                    metric = self.vector_tasks.metrics_out_queue.get(timeout=1)
                    count -= 1
                if isinstance(metric, tuple):
                    pkg_type, info = metric
                    if pkg_type in ["valid_metrics", "test_metrics"]:
                        mode = pkg_type.split("_")[0]
                        scalars, render = info
                        metrics = OrderedDict(
                            sorted(
                                [(k, v) for k, v in scalars.items()], key=lambda x: x[0]
                            )
                        )

                        message = ["{}".format(mode)]
                        add_step = True
                        for k in metrics:
                            if add_step:
                                metrics_steps = metrics[k][1]
                                message += ["{} steps:".format(metrics_steps)]
                                add_step = False
                            self.log_writer.add_scalar(
                                "{}/".format(mode) + k, metrics[k][0], metrics_steps,
                            )
                            message += [k + " {}".format(metrics[k][0])]
                        LOGGER.info(" ".join(message))

                        if render is not None:
                            self.log_writer.add_vid(
                                "{}/agent_view".format(mode), render, metrics_steps,
                            )
                    else:
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
                            losses.append(cscalars)
                        elif pkg_type == "teacher_package":
                            cscalars = {k: v for k, v in info.items()}
                            teachers.append(cscalars)
                        else:
                            LOGGER.warning("Unknown info package {}".format(info))
                else:
                    train_metrics.append(metric)
            except queue.Empty:
                pass

        for metric in train_metrics:
            self.scalars.add_scalars(
                OrderedDict(
                    sorted(
                        [(k, v) for k, v in metric.items() if k != "task_info"],
                        key=lambda x: x[0],
                    )
                )
            )
        for loss in losses:
            self.scalars.add_scalars(
                OrderedDict(
                    sorted(
                        [(k, v) for k, v in loss.items() if k != "task_info"],
                        key=lambda x: x[0],
                    )
                )
            )
        for teacher in teachers:
            self.scalars.add_scalars(
                OrderedDict(
                    sorted(
                        [(k, v) for k, v in teacher.items() if k != "task_info"],
                        key=lambda x: x[0],
                    )
                )
            )

        tracked_means = self.scalars.pop_and_reset()
        message = ["train {} steps:".format(self.total_steps + self.step_count)]
        for k in tracked_means:
            self.log_writer.add_scalar(
                "train/" + k, tracked_means[k], self.total_steps + self.step_count,
            )
            message += [k + " {}".format(tracked_means[k])]
        if len(tracked_means) > 0:
            LOGGER.info(" ".join(message))

    def update(self, rollouts: RolloutStorage) -> None:
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

                if self.lr_scheduler is not None:
                    info["lr"] = self.optimizer.param_groups[0]["lr"]  # type: ignore

                total_loss: Optional[torch.Tensor] = None
                for loss_name in self.losses:
                    loss, loss_weight = (
                        self.losses[loss_name],
                        self.loss_weights[loss_name],
                    )

                    current_loss, current_info = loss.loss(
                        step_count=self.step_count,
                        batch=batch,
                        actor_critic_output=actor_critic_output,
                    )
                    if total_loss is None:
                        total_loss = loss_weight * current_loss
                    else:
                        total_loss = total_loss + loss_weight * current_loss

                    info["losses"][loss_name] = current_info
                assert total_loss is not None, "No losses specified?"

                if isinstance(total_loss, torch.Tensor):
                    info["total_loss"] = total_loss.item()
                    self.vector_tasks.metrics_out_queue.put(("update_package", info))

                    self.optimizer.zero_grad()  # type: ignore
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm,  # type: ignore
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

    def collect_rollout_step(self, rollouts: RolloutStorage, render=None):
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

        rewards: Union[List, torch.Tensor]
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

        if render is not None and len(keep) > 0:
            render.append(self.vector_tasks.render(mode="rgb_array"))

        rollouts.reshape(keep)

        rollouts.insert(
            observations=self._preprocess_observations(batch)
            if len(keep) > 0
            else batch,
            recurrent_hidden_states=recurrent_hidden_states[:, keep],
            actions=actions[keep],
            action_log_probs=actor_critic_output.distributions.log_probs(actions)[keep],
            value_preds=actor_critic_output.values[keep],
            rewards=rewards[keep],
            masks=masks[keep],
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

    def initialize_rollouts(self, rollouts, render: Optional[List[np.ndarray]] = None):
        observations = self.vector_tasks.get_observations()
        npaused, keep, batch = self.remove_paused(observations)
        if render is not None and len(keep) > 0:
            render.append(self.vector_tasks.render(mode="rgb_array"))
        rollouts.reshape(keep)
        rollouts.to(self.device)
        rollouts.insert_initial_observations(
            self._preprocess_observations(batch) if len(keep) > 0 else batch
        )
        return npaused

    def train(self, rollouts: RolloutStorage):
        self.initialize_rollouts(rollouts)

        while self.rollout_count < self.num_rollouts:
            for step in range(self.steps_in_rollout):
                self.collect_rollout_step(rollouts)

            with torch.no_grad():
                step_observation = {k: v[-1] for k, v in rollouts.observations.items()}

                actor_critic_output, _ = self.actor_critic(
                    step_observation,
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.prev_actions[-1],
                    rollouts.masks[-1],
                )

            rollouts.compute_returns(
                actor_critic_output.values.detach(),
                self.use_gae,
                self.gamma,
                self.gae_lambda,
            )

            self.update(rollouts)
            rollouts.after_update()
            self.rollout_count += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch=self.step_count + self.total_steps)

            if (
                self.step_count - self.last_log >= self.log_interval
                or self.rollout_count == self.num_rollouts
            ):
                self.log()
                self.last_log = self.step_count

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

            if (self.advance_scene_rollout_period is not None) and (
                self.rollout_count % self.advance_scene_rollout_period == 0
            ):
                self.vector_tasks.next_task(force_advance_scene=True)
                self.initialize_rollouts(rollouts)

    def setup_stage(
        self,
        losses: Dict[str, AbstractActorCriticLoss],
        loss_weights: Dict[str, float],
        steps_in_rollout: int,
        stage_task_steps: int,
        update_epochs: int,
        update_mini_batches: int,
        gamma: float,
        use_gae: bool,
        gae_lambda: float,
        max_grad_norm: float,
        advance_scene_rollout_period: Optional[int] = None,
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
        message = "Using %d rollouts, %d steps (from requested %d steps)" % (
            self.num_rollouts,
            self.num_rollouts * self.num_processes * self.steps_in_rollout,
            self.stage_task_steps,
        )
        LOGGER.info(message)

        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.max_grad_norm = max_grad_norm

        self.advance_scene_rollout_period = advance_scene_rollout_period

        self.teacher_forcing = teacher_forcing

        self.deterministic_agent = deterministic_agent

    def _get_loss(self, loss_name) -> AbstractActorCriticLoss:
        assert (
            loss_name in self.training_pipeline.named_losses
        ), "undefined referenced loss"
        if isinstance(self.training_pipeline.named_losses[loss_name], Builder):
            return typing.cast(
                Builder[AbstractActorCriticLoss],
                self.training_pipeline.named_losses[loss_name],
            )()
        else:
            return typing.cast(
                AbstractActorCriticLoss, self.training_pipeline.named_losses[loss_name]
            )

    def _load_losses(self, stage: PipelineStage):
        stage_losses: Dict[str, AbstractActorCriticLoss] = {}
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

    def _stage_value(self, stage: PipelineStage, field: str, allow_none: bool = False):
        if hasattr(stage, field) and getattr(stage, field) is not None:
            return getattr(stage, field)

        if (
            hasattr(self.training_pipeline, field)
            and getattr(self.training_pipeline, field) is not None
        ):
            return getattr(self.training_pipeline, field)

        if field in self.machine_params:
            return self.machine_params[field]

        if allow_none:
            return None
        else:
            raise RuntimeError("missing value for {}".format(field))

    @property
    def log_writer_path(self) -> str:
        if self.extra_tag == "":
            return os.path.join(
                self.output_dir, "tb", self.experiment_name, self.local_start_time_str
            )
        else:
            return os.path.join(
                self.output_dir,
                "tb",
                self.experiment_name,
                self.extra_tag,
                self.local_start_time_str,
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

    def save_config_files(self):
        for file in self.loaded_config_src_files:
            base, module = self.loaded_config_src_files[file]
            parts = module.split(".")

            src_file = os.path.sep.join([base] + parts) + ".py"
            if not os.path.isfile(src_file):
                LOGGER.error("Config file {} not found".format(src_file))

            dst_file = (
                os.path.join(
                    self.configs_folder,
                    self.local_start_time_str,
                    os.path.join(*parts[1:]),
                )
                + ".py"
            )
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)

            shutil.copy(src_file, dst_file)

    def run_pipeline(self, checkpoint_file_name: Optional[str] = None):
        encountered_exception = False
        try:
            start_time = time.time()

            if checkpoint_file_name is not None:
                self.checkpoint_load(
                    self.get_checkpoint_path(checkpoint_file_name), verbose=True
                )

            self.local_start_time_str = time.strftime(
                "%Y-%m-%d_%H-%M-%S", time.localtime(start_time)
            )

            self.log_writer = SummaryWriter(log_dir=self.log_writer_path)

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
                    advance_scene_rollout_period=self._stage_value(
                        stage, "advance_scene_rollout_period"
                    ),
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
        except Exception as e:
            encountered_exception = True
            raise e
        finally:
            if not encountered_exception:
                LOGGER.info("\n\nTRAINING COMPLETE.\n\n")
            else:
                LOGGER.info("\n\nENCOUNTERED EXCEPTION DURING TRAINING!\n\n")
                LOGGER.exception(traceback.format_exc())
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
                    scalars, render, samples = self.run_eval(checkpoint_file_name=data)
                    write_to_parent.put(("valid_metrics", (scalars, render)))
                elif command in ["quit", "exit", "close"]:
                    self.close(verbose=False)
                    sys.exit()
                else:
                    raise NotImplementedError()

                new_data = False
        except KeyboardInterrupt:
            print("Eval KeyboardInterrupt")

    def process_video(self, render, max_clip_len=500):
        if len(render) > 0:
            nt = len(render)
            if nt > max_clip_len:
                LOGGER.info(
                    "Cutting video with length {} to {}".format(nt, max_clip_len)
                )
                render = render[:max_clip_len]
            try:
                render = np.stack(render, axis=0)  # T, H, W, C
                render = render.transpose((0, 3, 1, 2))  # T, C, H, W
                render = np.expand_dims(render, axis=0)  # 1, T, C, H, W
                render = tensor_to_video(render, fps=4)
            except MemoryError:
                LOGGER.warning(
                    "Skipped video with length {} (cut to {})".format(nt, max_clip_len)
                )
                render = None
        else:
            render = None
        return render

    def run_eval(self, checkpoint_file_name: str, rollout_steps=1, max_clip_len=2000):
        self.checkpoint_load(checkpoint_file_name, verbose=False)

        rollouts = RolloutStorage(
            rollout_steps,
            self.num_processes,
            self.actor_critic.action_space,
            self.actor_critic.recurrent_hidden_state_size,
            num_recurrent_layers=self.actor_critic.num_recurrent_layers,
        )

        render: Union[None, np.ndarray, List[np.ndarray]] = []
        num_paused = self.initialize_rollouts(rollouts, render=render)
        steps = 0
        while num_paused < self.num_processes:
            num_paused += self.collect_rollout_step(rollouts, render=render)
            steps += 1
            if steps % rollout_steps == 0:
                rollouts.after_update()

        self.vector_tasks.resume_all()
        self.vector_tasks.reset_all()

        render = self.process_video(render, max_clip_len)

        metrics, samples = self.process_eval_metrics(count=self.num_processes)

        return (
            {k: (v, self.total_steps + self.step_count) for k, v in metrics.items()},
            render,
            samples,
        )

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
        return (
            files[:: skip_checkpoints + 1]
            + (
                [files[-1]]
                if skip_checkpoints > 0 and len(files) % (skip_checkpoints + 1) != 1
                else []
            )
            if len(files) > 0
            else files
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

        os.makedirs(self.metric_path, exist_ok=True)
        fname = os.path.join(self.metric_path, "metrics" + suffix + ".json")

        LOGGER.info("Saving metrics in {}".format(fname))

        all_results = []
        for it, checkpoint_file_name in enumerate(checkpoints):
            step = self.step_from_checkpoint(checkpoint_file_name)
            LOGGER.info("{}/{} {} steps".format(it + 1, len(checkpoints), step,))

            scalars, render, samples = self.run_eval(
                checkpoint_file_name, rollout_steps
            )

            self.vector_tasks.metrics_out_queue.put(("test_metrics", (scalars, render)))

            results = {scalar: scalars[scalar][0] for scalar in scalars}
            results.update({"training_steps": step, "tasks": samples})
            all_results.append(results)

            with open(fname, "w") as f:
                json.dump(all_results, f, indent=4)

            self.log(count=1)

            with open(fname, "w") as f:
                json.dump(all_results, f, indent=4)

        LOGGER.info("Metrics saved in {}".format(fname))

    def close(self, verbose=True):
        if self._is_closed:
            return

        def logif(s: Union[str, Exception]):
            if verbose:
                if isinstance(s, str):
                    LOGGER.info(s)
                elif isinstance(s, Exception):
                    LOGGER.exception(traceback.format_exc())
                else:
                    raise NotImplementedError()

        try:
            logif("Closing OnPolicyRLEngine.vector_tasks.")
            self.vector_tasks.close()
            logif("Closed.")
        except Exception as e:
            logif("Exception raised when closing OnPolicyRLEngine.vector_tasks:")
            logif(e)
            pass

        logif("\n\n")
        try:
            logif("Closing OnPolicyRLEngine.eval_process")
            eval: mp.Process = getattr(self, "eval_process", None)
            if eval is not None:
                self.write_to_eval.put(("exit", None))
                eval.join(5)
                self.eval_process = None
            logif("Closed.")
        except Exception as e:
            logif("Exception raised when closing OnPolicyRLEngine.vector_tasks:")
            logif(e)
            pass

        logif("\n\n")
        try:
            logif("Closing OnPolicyRLEngine.log_writer")
            log_writer = getattr(self, "log_writer", None)
            if log_writer is not None:
                log_writer.close()
                self.log_writer = None
            logif("Closed.")
        except Exception as e:
            logif("Exception raised when closing OnPolicyRLEngine.log_writer:")
            logif(e)
            pass

        self._is_closed = True

    def __del__(self):
        self.close(verbose=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(verbose=False)


class OnPolicyTrainer(OnPolicyRLEngine):
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str,
        loaded_config_src_files: Optional[Dict[str, Tuple[str, str]]],
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        **kwargs
    ):
        super().__init__(
            config=config,
            loaded_config_src_files=loaded_config_src_files,
            output_dir=output_dir,
            seed=seed,
            mode="train",
            deterministic_cudnn=deterministic_cudnn,
            **kwargs,
        )


class OnPolicyValidator(OnPolicyRLEngine):
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        **kwargs
    ):
        super().__init__(
            config=config,
            loaded_config_src_files=None,
            output_dir=output_dir,
            seed=seed,
            mode="valid",
            deterministic_cudnn=deterministic_cudnn,
            **kwargs,
        )
        self.actor_critic.eval()


class OnPolicyTester(OnPolicyRLEngine):
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        **kwargs
    ):
        super().__init__(
            config=config,
            loaded_config_src_files=None,
            output_dir=output_dir,
            seed=seed,
            mode="test",
            deterministic_cudnn=deterministic_cudnn,
            **kwargs,
        )
        self.actor_critic.eval()
