from onpolicy_sync.storage import RolloutStorage
from rl_base.common import Loss
from typing import List, Optional, Any, Tuple
from onpolicy_sync.vector_task import VectorSampledTasks
from onpolicy_sync.model import Policy
import torch.optim
import torch
import torch.nn as nn
from onpolicy_sync.utils import batch_observations


class Trainer:
    def __init__(
        self,
        rollouts: RolloutStorage,
        vector_tasks: VectorSampledTasks,
        actor_critic: Policy,
        losses: List[Tuple[str, Loss, float]],
        optimizer: torch.optim.Optimizer,
        num_env_steps: int,
        update_epochs: int,
        update_mini_batches: int,
        gamma: float,
        use_gae: bool,
        gae_lambda: float,
        max_grad_norm: float,
        tracker: Any,
        teacher_forcing: Optional[torch.optim.lr_scheduler] = None,
    ):
        self.rollouts = rollouts
        self.vector_tasks = vector_tasks
        self.actor_critic = actor_critic

        self.losses = losses
        self.optimizer = optimizer

        self.num_env_steps = num_env_steps
        self.num_steps = self.rollouts.num_steps
        self.update_epochs = update_epochs
        self.update_mini_batches = update_mini_batches
        self.num_processes = self.vector_tasks._num_processes

        self.num_updates = (
            int(self.num_env_steps) // self.num_steps // self.num_processes
        )

        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.max_grad_norm = max_grad_norm

        self.tracker = tracker

        self.scheduler = teacher_forcing

        self.update_count = 0
        self.backprop_count = 0

    def update(self) -> None:
        advantages = self.rollouts.returns[:-1] - self.rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for e in range(self.update_epochs):
            data_generator = self.rollouts.recurrent_generator(
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

    def train(self):
        rollouts = self.rollouts
        actor_critic = self.actor_critic
        vtasks = self.vector_tasks

        observations = vtasks.next_task()
        rollouts.observations[0].copy_(batch_observations(observations))

        while self.update_count < self.num_updates:
            for step in range(self.num_steps):
                self.collect_rollout_step(rollouts)

            with torch.no_grad():
                step_observation = {k: v[-1] for k, v in rollouts.observations.items()}

                next_value = actor_critic.get_value(
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
        self.update_count += 1
