from onpolicy_sync.storage import RolloutStorage
from rl_base.common import Loss
from typing import List, Optional, Any, Tuple
from onpolicy_sync.vector_task import VectorSampledTasks
from onpolicy_sync.model import Policy
import torch.optim
import torch
import torch.nn as nn
from onpolicy_sync.utils import batch_observations
import os


class Trainer:
    def __init__(
        self,
        vector_tasks: VectorSampledTasks,
        actor_critic: Policy,
        losses: List[Tuple[str, Loss, float]],
        optimizer: torch.optim.Optimizer,
        num_steps: int,
        num_env_steps: int,
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
        teacher_forcing: Optional[torch.optim.lr_scheduler] = None,
    ):
        self.vector_tasks = vector_tasks
        self.actor_critic = actor_critic

        self.losses = losses
        self.optimizer = optimizer

        self.num_env_steps = num_env_steps
        self.num_steps = num_steps
        self.update_epochs = update_epochs
        self.update_mini_batches = update_mini_batches
        self.num_processes = num_processes

        self.num_updates = (
            int(self.num_env_steps) // self.num_steps // self.num_processes
        )

        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.max_grad_norm = max_grad_norm

        self.tracker = tracker

        self.scheduler = teacher_forcing

        self.models_folder = models_folder
        self.save_interval = save_interval

        self.update_count = 0
        self.backprop_count = 0

    def checkpoint_save(self) -> None:
        # save for every interval-th episode or for the last epoch
        if (
            self.save_interval > 0
            and (
                self.update_count % self.save_interval == 0
                or self.update_count == self.num_updates - 1
            )
            and self.models_folder != ""
        ):
            os.makedirs(self.models_folder, exist_ok=True)

            model_path = os.path.join(
                self.models_folder, "model_%010d.pt" % self.update_count
            )
            torch.save(
                {
                    "update_count": self.update_count,
                    "backprop_count": self.backprop_count,
                    "model_state_dict": self.actor_critic.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                model_path,
            )

    def checkpoint_load(self, checkpoint_file_name: str) -> None:
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = torch.load(checkpoint_file_name, map_location="cpu")
        self.actor_critic.load_state_dict(ckpt_dict["model_state_dict"])
        self.update_count = ckpt_dict["update_count"]
        self.backprop_count = ckpt_dict["backprop_count"]
        self.optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])

    def update(self, rollouts) -> None:
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for e in range(self.update_epochs):
            data_generator = rollouts.recurrent_generator(
                advantages, self.update_mini_batches
            )

            for batch in data_generator:
                # Reshape to do in a single forward pass for all steps
                actor_critic_output = self.actor_critic.evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["masks"],
                    batch["actions"],
                )

                info = dict(
                    backprop_count=self.backprop_count,
                    update_count=self.update_count,
                    losses=[],
                )
                self.optimizer.zero_grad()
                for loss_name, loss, loss_weight in self.losses:
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

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
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
            actions_log_probs,
            values,
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

        while self.update_count < self.num_updates:
            for step in range(self.num_steps):
                self.collect_rollout_step(rollouts)

            with torch.no_grad():
                step_observation = {k: v[-1] for k, v in rollouts.observations.items()}

                next_value = self.actor_critic.get_value(
                    step_observation,
                    rollouts.recurrent_hidden_states[-1],
                    rollouts.prev_actions[-1],
                    rollouts.masks[-1],
                ).detach()

            rollouts.compute_returns(
                next_value, self.use_gae, self.gamma, self.gae_lambda,
            )

            self.update()

            rollouts.after_update()

        if self.scheduler is not None:
            self.scheduler.step()

        self.checkpoint_save()

        self.update_count += 1
