from typing import Optional, Any, Dict, Union
import os
from collections import deque
import queue

import torch.optim
import torch
import torch.distributions
from tensorboardX import SummaryWriter

from onpolicy_sync.vector_task import VectorSampledTasks
from onpolicy_sync.utils import batch_observations, ScalarMeanTracker
from rl_base.experiment_config import ExperimentConfig
from configs.util import Builder
from rl_base.preprocessor import ObservationSet
from onpolicy_sync.storage import RolloutStorage


def evaluate(config: ExperimentConfig, output_dir: str, checkpoint_file_name: str):
    evaluator = Evaluator(config, output_dir)
    evaluator.run_eval(checkpoint_file_name)


class Evaluator:
    def __init__(self, config: ExperimentConfig, output_dir: str):
        self.evaluation_params = config.evaluation_params()

        evaluation_params = self.evaluation_params

        self.device = "cpu"
        if len(evaluation_params["gpu_ids"]) > 0:
            if not torch.cuda.is_available():
                print(
                    "Warning: no CUDA devices available for gpu ids {}".format(
                        evaluation_params["gpu_ids"]
                    )
                )
            else:
                self.device = "cuda:%d" % evaluation_params["gpu_ids"][0]
                torch.cuda.set_device(self.device)

        self.observation_set = None
        if "observation_set" in evaluation_params:
            all_preprocessors = []
            sensor_ids = []
            preprocessor_ids = []
            for observation in evaluation_params["observation_set"]:
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
            self.observation_set = ObservationSet(
                sensor_ids, preprocessor_ids, all_preprocessors
            )

        self.actor_critic = config.create_model().to(self.device)

        devices = (
            evaluation_params["sampler_devices"]
            if "sampler_devices" in evaluation_params
            else evaluation_params["gpu_ids"]
        )
        sampler_fn_args = [
            config.valid_task_sampler_args(
                process_ind=it,
                total_processes=evaluation_params["nprocesses"],
                devices=devices,
            )
            for it in range(evaluation_params["nprocesses"])
        ]

        self.vector_tasks = VectorSampledTasks(
            make_sampler_fn=config.make_sampler_fn, sampler_fn_args=sampler_fn_args
        )

        self.tracker = deque()

        self.output_folder = output_dir
        self.eval_folder = os.path.join(self.output_folder, "eval")
        os.makedirs(self.eval_folder, exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=self.eval_folder)
        self.scalars = ScalarMeanTracker()

        self.step_count = 0
        self.total_steps = 0

        self.deterministic = True

        self.num_processes = evaluation_params["nprocesses"]

    def checkpoint_load(self, ckpt: Union[str, Dict[str, Any]]) -> None:
        if isinstance(ckpt, str):
            # Map location CPU is almost always better than mapping to a CUDA device.
            ckpt = torch.load(ckpt, map_location="cpu")

        self.actor_critic.load_state_dict(ckpt["model_state_dict"])
        self.step_count = ckpt["step_count"]
        self.total_steps = ckpt["total_steps"]

    def log(self):
        while not self.vector_tasks.metrics_out_queue.empty():
            try:
                metric = self.vector_tasks.metrics_out_queue.get_nowait()
                self.scalars.add_scalars(metric)
            except queue.Empty:
                pass

        tracked_means = self.scalars.pop_and_reset()
        for k in tracked_means:
            self.log_writer.add_scalar(
                "eval/" + k, tracked_means[k], self.total_steps + self.step_count
            )

    def _preprocess_observations(self, batched_observations):
        if self.observation_set is None:
            return batched_observations
        return self.observation_set.get_observations(batched_observations)

    def collect_rollout_step(self, rollouts):
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
            if not self.deterministic
            else actor_critic_output.distributions.mode()
        )

        outputs = self.vector_tasks.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        rewards = torch.tensor(rewards, dtype=torch.float)
        rewards = rewards.unsqueeze(1)

        # If done then clean the history of observations.
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float32
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

    def initialize_rollouts(self, rollouts):
        observations = self.vector_tasks.get_observations()
        npaused, keep, batch = self.remove_paused(observations)
        rollouts.reshape(keep)
        rollouts.insert_initial_observations(self._preprocess_observations(batch))
        rollouts.to(self.device)
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

    def run_eval(self, checkpoint_file_name: str):
        self.checkpoint_load(checkpoint_file_name)

        rollouts = RolloutStorage(
            1,
            self.num_processes,
            self.actor_critic.action_space,
            self.actor_critic.recurrent_hidden_state_size,
        )

        num_paused = self.initialize_rollouts(rollouts)
        steps = 0
        while num_paused < self.num_processes:
            num_paused += self.collect_rollout_step(rollouts)
            steps += 1

        self.log()
        self.close()

    def close(self):
        self.vector_tasks.close()
        self.log_writer.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
