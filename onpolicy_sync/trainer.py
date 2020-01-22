from typing import Optional, Any, Dict, Union
import os
from collections import deque
import queue

import torch.optim
import torch
import torch.nn as nn
import torch.distributions
from tensorboardX import SummaryWriter

from imitation.utils import LinearDecay
from rl_base.common import Loss
from onpolicy_sync.vector_task import VectorSampledTasks
from onpolicy_sync.policy import ActorCriticModel
from onpolicy_sync.utils import batch_observations, ScalarMeanTracker
from rl_base.distributions import CategoricalDistr
from rl_base.experiment_config import ExperimentConfig
from configs.util import Builder
from rl_base.preprocessor import ObservationSet
from onpolicy_sync.storage import RolloutStorage


class Trainer:
    def __init__(self, config: ExperimentConfig, output_dir: str):
        self.train_pipeline = config.training_pipeline()

        train_pipeline = self.train_pipeline

        self.device = "cpu"
        if torch.cuda.is_available() and len(train_pipeline["gpu_ids"]) > 0:
            self.device = "cuda:%d" % train_pipeline["gpu_ids"][0]
            torch.cuda.set_device(self.device)

        self.observation_set = None
        if "observation_set" in train_pipeline:
            all_preprocessors = []
            sensor_ids = []
            preprocessor_ids = []
            for observation in train_pipeline["observation_set"]:
                if isinstance(observation, str):
                    sensor_ids.append(observation)
                else:
                    if isinstance(observation, Builder):
                        all_preprocessors.append(
                            observation(config={"device": self.device})
                        )
                    else:
                        all_preprocessors.append(observation)
                    preprocessor_ids.append(all_preprocessors[-1].uuid)
            print(
                "sensors in obs", sensor_ids, "preprocessors in obs", preprocessor_ids
            )
            self.observation_set = ObservationSet(
                sensor_ids, preprocessor_ids, all_preprocessors
            )
            print("created observation set")

        self.actor_critic = config.create_model().to(self.device)

        self.optimizer = train_pipeline["optimizer"]
        if isinstance(self.optimizer, Builder):
            self.optimizer = self.optimizer(
                params=[p for p in self.actor_critic.parameters() if p.requires_grad]
            )

        sampler_fn_args = [
            config.train_task_sampler_args(
                process_ind=it, total_processes=train_pipeline["nprocesses"]
            )
            for it in range(train_pipeline["nprocesses"])
        ]

        self.vector_tasks = VectorSampledTasks(
            make_sampler_fn=config.make_sampler_fn, sampler_fn_args=sampler_fn_args
        )

        self.tracker = deque()

        self.models_folder = os.path.join(output_dir, "models")
        os.makedirs(self.models_folder, exist_ok=True)

        self.log_writer = SummaryWriter(log_dir=output_dir)
        self.scalars = ScalarMeanTracker()

        self.total_updates = 0
        self.pipeline_stage = 0

        self.save_interval = train_pipeline["save_interval"]
        self.log_interval = train_pipeline["log_interval"]
        self.num_processes = train_pipeline["nprocesses"]

    def checkpoint_save(self) -> None:
        os.makedirs(self.models_folder, exist_ok=True)

        model_path = os.path.join(
            self.models_folder,
            "model_stage_%02d_%010d.pt" % (self.pipeline_stage, self.rollout_count),
        )
        torch.save(
            {
                "total_updates": self.total_updates,
                "pipeline_stage": self.pipeline_stage,
                "rollout_count": self.rollout_count,
                "backprop_count": self.backprop_count,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_state_dict": self.actor_critic.state_dict(),
            },
            model_path,
        )

    def checkpoint_load(self, ckpt: Union[str, Dict[str, Any]]) -> None:
        if isinstance(ckpt, str):
            print("Loading checkpoint from %s" % ckpt)
            # Map location CPU is almost always better than mapping to a CUDA device.
            ckpt = torch.load(ckpt, map_location="cpu")

        self.actor_critic.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.backprop_count = ckpt["backprop_count"]
        self.rollout_count = ckpt["rollout_count"]
        self.pipeline_stage = ckpt["pipeline_stage"]
        self.total_updates = ckpt["total_updates"]
        self.train_pipeline["pipeline"] = self.train_pipeline["pipeline"][
            self.pipeline_stage :
        ]

    def log(self):
        if len(self.tracker) < self.log_interval:
            return

        while len(self.tracker):
            info = self.tracker.popleft()
            # print(info)
            cscalars = {}
            for loss in info["losses"]:
                lossname = loss[:-5] if loss.endswith("_loss") else loss
                for scalar in info["losses"][loss]:
                    cscalars["/".join([lossname, scalar])] = info["losses"][loss][
                        scalar
                    ]
            self.scalars.add_scalars(cscalars)

        while not self.vector_tasks.metrics_out_queue.empty():
            # process metrics without blocking
            try:
                metric = self.vector_tasks.metrics_out_queue.get_nowait()
                # print(metric)
                self.scalars.add_scalars(metric)
            except queue.Empty:
                pass

        tracked_means = self.scalars.pop_and_reset()
        for k in tracked_means:
            self.log_writer.add_scalar(
                "train/" + k, tracked_means[k], self.total_updates + self.rollout_count
            )

    def update(self, rollouts) -> None:
        # print("new update")
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        for e in range(self.update_epochs):
            data_generator = rollouts.recurrent_generator(
                advantages, self.update_mini_batches
            )

            for bit, batch in enumerate(data_generator):
                batch = {
                    k: batch[k].to(self.device) if k != "observations" else batch[k]
                    for k in batch
                }
                # Reshape to do in a single forward pass for all steps
                batch["observations"] = {
                    k: v.to(self.device) for k, v in batch["observations"].items()
                }
                actor_critic_output, hidden_states = self.actor_critic(
                    batch["observations"],
                    batch["recurrent_hidden_states"].to(self.device),
                    batch["prev_actions"].to(self.device),
                    batch["masks"].to(self.device),
                )

                info = dict(
                    total_updates=self.total_updates,
                    backprop_count=self.backprop_count,
                    rollout_count=self.rollout_count,
                    epoch=e,
                    batch=bit,
                    losses={},
                )
                self.optimizer.zero_grad()
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
                        total_loss += loss_weight * current_loss

                    info["losses"][loss_name] = current_info
                assert total_loss is not None, "No losses specified?"
                self.tracker.append(info)

                # print(info)

                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.backprop_count += 1

    def collect_rollout_step(self, rollouts):
        # print("new rollout step")
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step].to(self.device)
                for k, v in rollouts.observations.items()
            }

            actor_critic_output, recurrent_hidden_states = self.actor_critic(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step].to(self.device),
                rollouts.prev_actions[rollouts.step].to(self.device),
                rollouts.masks[rollouts.step].to(self.device),
            )

        actions = (
            actor_critic_output.distributions.sample()
            if not self.deterministic
            else actor_critic_output.distributions.mode()
        )
        if (
            self.teacher_forcing is not None
            and self.teacher_forcing(self.rollout_count) > 0
        ):
            tf_mask_shape = step_observation["expert_action"].shape[:-1] + (1,)
            expert_actions = (
                step_observation["expert_action"].view(-1, 2)[:, 0].view(*tf_mask_shape)
            )
            expert_action_exists_mask = (
                step_observation["expert_action"].view(-1, 2)[:, 1].view(*tf_mask_shape)
            )
            teacher_forcing_mask = (
                torch.distributions.bernoulli.Bernoulli(
                    torch.tensor(self.teacher_forcing(self.rollout_count))
                )
                .sample(tf_mask_shape)
                .long()
                .to(self.device)
            ) * expert_action_exists_mask
            actions = (
                teacher_forcing_mask * expert_actions
                + (1 - teacher_forcing_mask) * actions
            )

        outputs = self.vector_tasks.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        batch = batch_observations(observations)
        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        # If done then clean the history of observations.
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float32
        )

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actor_critic_output.distributions.log_probs(actions),
            actor_critic_output.values,
            rewards,
            masks,
        )

    def initialize_rollouts(self, rollouts):
        # print("initialize rollouts")
        observations = self.vector_tasks.get_observations()
        batch = batch_observations(observations)

        rollouts.insert_initial_observations(batch)

    def train(self, rollouts):
        self.initialize_rollouts(rollouts)

        while self.rollout_count < self.num_rollouts:
            # print("new rollout")
            for step in range(self.steps_in_rollout):
                self.collect_rollout_step(rollouts)

            with torch.no_grad():
                step_observation = {
                    k: v[-1].to(self.device) for k, v in rollouts.observations.items()
                }

                actor_critic_output, _ = self.actor_critic(
                    step_observation,
                    rollouts.recurrent_hidden_states[-1].to(self.device),
                    rollouts.prev_actions[-1].to(self.device),
                    rollouts.masks[-1].to(self.device),
                )

            rollouts.compute_returns(
                actor_critic_output.values, self.use_gae, self.gamma, self.gae_lambda,
            )

            self.update(rollouts)

            rollouts.after_update()

            self.log()

            self.rollout_count += 1

            # save for every interval-th episode or for the last epoch
            if (
                self.save_interval > 0
                and (
                    self.rollout_count % self.save_interval == 0
                    or self.rollout_count == self.num_rollouts
                )
                and self.models_folder != ""
            ):
                self.checkpoint_save()

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
        deterministic: bool = False,
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
        print("Using %d rollouts" % self.num_rollouts)

        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.max_grad_norm = max_grad_norm

        self.teacher_forcing = teacher_forcing

        self.deterministic = deterministic

    def _get_loss(self, current_loss):
        assert current_loss in self.train_pipeline, "undefined referenced loss"
        if isinstance(self.train_pipeline[current_loss], Builder):
            return self.train_pipeline[current_loss](optimizer=self.optimizer)
        else:
            return self.train_pipeline[current_loss]

    def _load_losses(self, stage):
        stage_losses = dict()
        for current_loss in stage["losses"]:
            stage_losses[current_loss] = self._get_loss(current_loss)

        stage_weights = {name: 1.0 for name in stage["losses"]}
        for current_loss in self.train_pipeline.get("loss_weights", []):
            if current_loss in stage_losses:
                stage_weights[current_loss] = self.train_pipeline["loss_weights"][
                    current_loss
                ]
        for current_loss in stage.get("loss_weights", []):
            assert current_loss in stage_losses, (
                "missing loss definition for weight %s" % current_loss
            )
            stage_weights[current_loss] = stage["loss_weights"][current_loss]

        return stage_losses, stage_weights

    def _stage_value(self, stage, field):
        assert field in stage or field in self.train_pipeline, (
            "missing value for %s" % field
        )
        return stage[field] if field in stage else self.train_pipeline[field]

    def run_pipeline(self, checkpoint_file_name: Optional[str] = None):
        if checkpoint_file_name is not None:
            self.checkpoint_load(checkpoint_file_name)

        for stage in self.train_pipeline["pipeline"]:
            stage_limit = stage["end_criterion"]
            stage_losses, stage_weights = self._load_losses(stage)

            self.setup_stage(
                losses=stage_losses,
                loss_weights=stage_weights,
                steps_in_rollout=self._stage_value(stage, "num_steps"),
                stage_task_steps=stage_limit,
                update_epochs=self._stage_value(stage, "update_repeats"),
                update_mini_batches=self._stage_value(stage, "num_mini_batch"),
                gamma=self._stage_value(stage, "gamma"),
                use_gae=self._stage_value(stage, "use_gae"),
                gae_lambda=self._stage_value(stage, "gae_lambda"),
                max_grad_norm=self._stage_value(stage, "max_grad_norm"),
                teacher_forcing=stage.get("teacher_forcing"),
            )

            self.train(
                RolloutStorage(
                    self.steps_in_rollout,
                    self.num_processes,
                    self.actor_critic.action_space,
                    self.actor_critic.recurrent_hidden_state_size,
                    observation_set=self.observation_set,
                )
            )

            self.total_updates += self.num_rollouts
            self.pipeline_stage += 1

            self.rollout_count = 0
            self.backprop_count = 0
