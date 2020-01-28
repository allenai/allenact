import os
import queue
import shutil
import time
import typing
from typing import Optional, Any, Dict, Union

import torch
import torch.distributions
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
from tensorboardX import SummaryWriter


from configs.util import Builder
from onpolicy_sync.storage import RolloutStorage
from onpolicy_sync.utils import LinearDecay
from onpolicy_sync.utils import batch_observations, ScalarMeanTracker
from onpolicy_sync.vector_task import VectorSampledTasks
from rl_base.common import Loss
from rl_base.experiment_config import ExperimentConfig
from rl_base.preprocessor import ObservationSet


def validate(
    config: ExperimentConfig,
    output_dir: str,
    read_from_parent: mp.Queue,
    write_to_parent: mp.Queue,
    seed: Optional[int] = None,
    disable_cudnn: bool = False,
):
    evaluator = Trainer(
        config, None, output_dir, mode="valid", seed=seed, disable_cudnn=disable_cudnn,
    )
    evaluator.process_checkpoints(read_from_parent, write_to_parent)


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        loaded_config_src_files: Optional[Dict[str, str]],
        output_dir: str,
        seed: Optional[int] = None,
        mode: str = "train",
        disable_cudnn: bool = False,
    ):
        self.disable_cudnn = disable_cudnn
        self.seed = seed
        self.mode = mode.lower()
        assert self.mode in [
            "train",
            "valid",
            "test",
        ], "Only train, valid, test modes supported"

        self.params = self.get_params(config)

        self.device = "cpu"
        if len(self.params["gpu_ids"]) > 0:
            if not torch.cuda.is_available():
                print(
                    "Warning: no CUDA devices available for gpu ids {}".format(
                        self.params["gpu_ids"]
                    )
                )
            else:
                self.device = "cuda:%d" % self.params["gpu_ids"][0]
                torch.cuda.set_device(self.device)  # type: ignore

        if self.seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                if self.disable_cudnn:
                    # torch.backends.cudnn.deterministic = True
                    # torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.enabled = False

        self.observation_set = None
        if "preprocessors" in self.params and "observation_set" in self.params:
            all_preprocessors = []
            for preprocessor in self.params["preprocessors"]:
                if isinstance(preprocessor, Builder):
                    all_preprocessors.append(
                        preprocessor(config={"device": self.device})
                    )
                else:
                    all_preprocessors.append(preprocessor)

            self.observation_set = ObservationSet(
                self.params["observation_set"], all_preprocessors
            )

        self.actor_critic = config.create_model().to(self.device)

        self.optimizer = None
        if "optimizer" in self.params:
            self.optimizer = self.params["optimizer"]
            if isinstance(self.optimizer, Builder):
                self.optimizer = self.optimizer(
                    params=[
                        p for p in self.actor_critic.parameters() if p.requires_grad
                    ]
                )

        self.vector_tasks = VectorSampledTasks(
            make_sampler_fn=config.make_sampler_fn,
            sampler_fn_args=self.get_sampler_fn_args(config),
        )

        self.output_dir = output_dir
        self.models_folder = os.path.join(output_dir, "models")
        os.makedirs(self.models_folder, exist_ok=True)

        self.configs_folder = os.path.join(output_dir, "configs")
        os.makedirs(self.configs_folder, exist_ok=True)
        if mode == "train":
            for file in loaded_config_src_files:
                parts = loaded_config_src_files[file].split(".")
                src_file = os.path.sep.join(parts) + ".py"
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
        self.deterministic: Optional[bool] = None
        self.local_start_time_str: Optional[str] = None

        self.experiment_name = config.tag()

        self.save_interval = self.params["save_interval"]
        self.log_interval = self.params["log_interval"]
        self.num_processes = self.params["nprocesses"]

        self.config = config

        self.write_to_eval = None
        if self.mode == "train":
            self.mp_ctx = self.vector_tasks.mp_ctx
            self.write_to_eval = self.mp_ctx.Queue()
            self.eval_process = self.mp_ctx.Process(
                target=validate,
                args=(
                    self.config,
                    self.output_dir,
                    self.write_to_eval,
                    self.vector_tasks.metrics_out_queue,
                    self.seed,
                    self.disable_cudnn,
                ),
            )
            self.eval_process.start()

    def get_params(self, config: ExperimentConfig):
        if self.mode == "train":
            return config.training_pipeline()
        elif self.mode == "valid":
            return config.evaluation_params()
        elif self.mode == "test":
            return config.evaluation_params()

    def get_sampler_fn_args(self, config: ExperimentConfig):
        devices = (
            self.params["sampler_devices"]
            if "sampler_devices" in self.params
            else self.params["gpu_ids"]
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
                total_processes=self.params["nprocesses"],
                devices=devices,
                seed=self.seed,
            )
            for it in range(self.params["nprocesses"])
        ]

    def checkpoint_save(self) -> str:
        os.makedirs(self.models_folder, exist_ok=True)

        model_path = os.path.join(
            self.models_folder,
            "exp_{}__time_{}__stage_{}__steps_{}__seed_{}.pt".format(
                self.experiment_name,
                self.local_start_time_str,
                self.pipeline_stage,
                self.total_steps + self.step_count,
                self.seed,
            ),
        )
        torch.save(
            {
                "total_updates": self.total_updates,
                "total_steps": self.total_steps,
                "pipeline_stage": self.pipeline_stage,
                "rollout_count": self.rollout_count,
                "backprop_count": self.backprop_count,
                "step_count": self.step_count,
                "local_start_time_str": self.local_start_time_str,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_state_dict": self.actor_critic.state_dict(),
            },
            model_path,
        )
        return model_path

    def checkpoint_load(self, ckpt: Union[str, Dict[str, Any]]) -> None:
        if isinstance(ckpt, str):
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
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.backprop_count = ckpt["backprop_count"]  # type: ignore
            self.rollout_count = ckpt["rollout_count"]  # type: ignore
            self.pipeline_stage = ckpt["pipeline_stage"]  # type: ignore
            self.total_updates = ckpt["total_updates"]  # type: ignore
            self.local_start_time_str = typing.cast(str, ckpt["local_start_time_str"])
            self.params["pipeline"] = self.params["pipeline"][self.pipeline_stage :]

    def process_valid_metrics(self):
        while not self.vector_tasks.metrics_out_queue.empty():
            try:
                metric = self.vector_tasks.metrics_out_queue.get_nowait()
                self.scalars.add_scalars(metric)
            except queue.Empty:
                pass

        return self.scalars.pop_and_reset()

    def log(self):
        valid_metrics = None
        while not self.vector_tasks.metrics_out_queue.empty():
            try:
                metric = self.vector_tasks.metrics_out_queue.get_nowait()
                if isinstance(metric, tuple):
                    pkg_type, info = metric
                    if pkg_type == "valid_metrics":
                        valid_metrics = {k: v for k, v in info.items()}
                    else:
                        cscalars: Optional[Dict[str, Union[float, int]]] = None
                        if pkg_type == "update_package":
                            cscalars = {"total_loss": info["total_loss"]}
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
            except queue.Empty:
                pass

        tracked_means = self.scalars.pop_and_reset()
        for k in tracked_means:
            self.log_writer.add_scalar(
                "train/" + k, tracked_means[k], self.total_steps + self.step_count,
            )

        if valid_metrics is not None:
            for k in valid_metrics:
                self.log_writer.add_scalar(
                    "valid/" + k, valid_metrics[k][0], valid_metrics[k][1],
                )

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
                        total_loss = total_loss + loss_weight * current_loss

                    info["losses"][loss_name] = current_info
                assert total_loss is not None, "No losses specified?"
                info["total_loss"] = total_loss.item()
                self.vector_tasks.metrics_out_queue.put(("update_package", info))

                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.backprop_count += 1

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
            if not self.deterministic
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
                    self.step_count % self.save_interval == 0
                    or self.rollout_count == self.num_rollouts
                )
                and self.models_folder != ""
            ):
                model_path = self.checkpoint_save()
                self.write_to_eval.put(("eval", model_path))

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
        print(
            "Using %d rollouts, %d steps (from %d)"
            % (
                self.num_rollouts,
                self.num_rollouts * self.num_processes * self.steps_in_rollout,
                self.stage_task_steps,
            )
        )

        self.gamma = gamma
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

        self.max_grad_norm = max_grad_norm

        self.teacher_forcing = teacher_forcing

        self.deterministic = deterministic

    def _get_loss(self, current_loss):
        assert current_loss in self.params, "undefined referenced loss"
        if isinstance(self.params[current_loss], Builder):
            return self.params[current_loss](optimizer=self.optimizer)
        else:
            return self.params[current_loss]

    def _load_losses(self, stage):
        stage_losses = dict()
        for current_loss in stage["losses"]:
            stage_losses[current_loss] = self._get_loss(current_loss)

        stage_weights = {name: 1.0 for name in stage["losses"]}
        for current_loss in self.params.get("loss_weights", []):
            if current_loss in stage_losses:
                stage_weights[current_loss] = self.params["loss_weights"][current_loss]
        for current_loss in stage.get("loss_weights", []):
            assert current_loss in stage_losses, (
                "missing loss definition for weight %s" % current_loss
            )
            stage_weights[current_loss] = stage["loss_weights"][current_loss]

        return stage_losses, stage_weights

    def _stage_value(self, stage, field):
        assert field in stage or field in self.params, "missing value for %s" % field
        return stage[field] if field in stage else self.params[field]

    def run_pipeline(self, checkpoint_file_name: Optional[str] = None):
        self.log_writer = SummaryWriter(log_dir=self.output_dir)

        start_time = time.time()
        self.local_start_time_str = time.strftime(
            "%Y-%m-%d_%H-%M-%S", time.localtime(start_time)
        )
        if checkpoint_file_name is not None:
            self.checkpoint_load(checkpoint_file_name)

        for stage in self.params["pipeline"]:
            self.last_log = self.step_count - self.log_interval

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
                )
            )

            self.total_updates += self.num_rollouts
            self.pipeline_stage += 1

            self.rollout_count = 0
            self.backprop_count = 0
            self.total_steps += self.step_count
            self.step_count = 0

        self.close()

    def process_checkpoints(
        self,
        read_from_parent: mp.Queue,
        write_to_parent: mp.Queue,
        deterministic: bool = True,
    ):
        self.deterministic = deterministic
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
                        pass

                if command == "eval":
                    scalars = self.run_eval(checkpoint_file_name=data)
                    write_to_parent.put(("valid_metrics", scalars))
                elif command == "close":
                    self.close()
                else:
                    raise NotImplementedError()

                new_data = False
        except KeyboardInterrupt:
            print("Eval KeyboardInterrupt - closing")
            self.close()

    def run_eval(self, checkpoint_file_name: str, rollout_steps=1):
        self.checkpoint_load(checkpoint_file_name)

        rollouts = RolloutStorage(
            rollout_steps,
            self.num_processes,
            self.actor_critic.action_space,
            self.actor_critic.recurrent_hidden_state_size,
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
            for k, v in self.process_valid_metrics().items()
        }

    def close(self):
        queue = getattr(self, "write_to_eval", None)
        if queue is not None:
            queue.put(("close",))
            self.write_to_eval = None
        eval = getattr(self, "eval_process", None)
        if eval is not None:
            eval.join()
        log_writer = getattr(self, "log_writer", None)
        if log_writer is not None:
            log_writer.close()
            self.log_writer = None
        queue = getattr(self, "write_to_eval", None)
        if queue is not None:
            queue.put(("close",))
            self.write_to_eval = None
        tasks = getattr(self, "vector_tasks", None)
        if tasks is not None:
            tasks.close()
            self.vector_tasks = None
        eval = getattr(self, "eval_process", None)
        if eval is not None:
            eval.join()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
