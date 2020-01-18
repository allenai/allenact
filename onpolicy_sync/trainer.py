from rl_base.common import Loss
from typing import Optional, Any, Dict
from onpolicy_sync.vector_task import VectorSampledTasks
from onpolicy_sync.policy import ActorCriticModel
import torch.optim
import torch
import torch.nn as nn
from onpolicy_sync.utils import batch_observations
import os
from rl_base.distributions import CategoricalDistr


class Trainer:
    def __init__(
        self,
        vector_tasks: VectorSampledTasks,
        actor_critic: ActorCriticModel[CategoricalDistr],
        losses: Dict[str, Loss],
        loss_weights: Dict[str, float],
        optimizer: torch.optim.Optimizer,
        steps_in_rollout: int,
        stage_task_steps: int,
        update_epochs: int,
        update_mini_batches: int,
        num_processes: int,
        gamma: float,
        use_gae: bool,
        gae_lambda: float,
        max_grad_norm: float,
        tracker: Any,
        models_folder: str,
        save_interval: int,
        pipeline_stage: int,
        teacher_forcing: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        deterministic: bool = False,
    ):
        self.vector_tasks = vector_tasks
        self.actor_critic = actor_critic

        self.losses = losses
        self.loss_weights = loss_weights
        self.optimizer = optimizer

        self.stage_task_steps = stage_task_steps
        self.steps_in_rollout = steps_in_rollout
        self.update_epochs = update_epochs
        self.update_mini_batches = update_mini_batches
        self.num_processes = num_processes

        self.num_rollouts = (
            int(self.stage_task_steps) // self.steps_in_rollout
        ) // self.num_processes

        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.max_grad_norm = max_grad_norm

        self.tracker = tracker

        self.teacher_forcing = teacher_forcing

        self.models_folder = models_folder
        self.save_interval = save_interval

        self.pipeline_stage = pipeline_stage

        self.deterministic = deterministic

        self.rollout_count = 0
        self.backprop_count = 0

    def checkpoint_save(self) -> None:
        os.makedirs(self.models_folder, exist_ok=True)

        model_path = os.path.join(
            self.models_folder,
            "model_stage_%02d_%010d.pt" % (self.pipeline_stage, self.rollout_count),
        )
        torch.save(
            {
                "pipeline_stage": self.pipeline_stage,
                "rollout_count": self.rollout_count,
                "backprop_count": self.backprop_count,
                "model_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            model_path,
        )

    def checkpoint_load(self, ckpt_dict: Dict[str, Any]) -> None:
        self.actor_critic.load_state_dict(ckpt_dict["model_state_dict"])
        self.rollout_count = ckpt_dict["rollout_count"]
        self.backprop_count = ckpt_dict["backprop_count"]
        self.optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])

    def update(self, rollouts) -> None:
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        for e in range(self.update_epochs):
            data_generator = rollouts.recurrent_generator(
                advantages, self.update_mini_batches
            )

            for bit, batch in enumerate(data_generator):
                # Reshape to do in a single forward pass for all steps
                actor_critic_output = self.actor_critic(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],  # for step 0, noop action as input
                    batch["masks"],
                )

                info = dict(
                    backprop_count=self.backprop_count,
                    rollout_count=self.rollout_count,
                    epoch=e,
                    batch=bit,
                    losses=[],
                )
                self.optimizer.zero_grad()
                for loss_name in self.losses:
                    loss, loss_weight = (
                        self.losses[loss_name],
                        self.loss_weights[loss_name],
                    )

                    current_loss, current_info = loss.loss(batch, actor_critic_output)
                    (loss_weight * current_loss).backward()

                    current_info["name"] = loss_name
                    current_info["weight"] = loss_weight
                    info["losses"].append(current_info)

                self.tracker.append(info)

                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.backprop_count += 1

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
            if self.deterministic
            else actor_critic_output.distributions.mode()
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
        observations = self.vector_tasks.next_task()
        batch = batch_observations(observations)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

    def train(self, rollouts):
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
                actor_critic_output.values, self.use_gae, self.gamma, self.gae_lambda,
            )

            self.update(rollouts)

            rollouts.after_update()

        if self.teacher_forcing is not None:
            self.teacher_forcing.step()

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
