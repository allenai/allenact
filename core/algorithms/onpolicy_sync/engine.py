"""Defines the reinforcement learning `OnPolicyRLEngine`."""
import itertools
import os
import queue
import random
import time
import traceback
from collections import defaultdict
from multiprocessing.context import BaseContext
from typing import (
    Optional,
    Any,
    Dict,
    Union,
    List,
    Sequence,
    cast,
    Iterator,
    Callable,
)

import torch
import torch.distributed as dist  # type: ignore
import torch.distributions  # type: ignore
import torch.multiprocessing as mp  # type: ignore
import torch.optim
from torch import nn
from torch import optim

# noinspection PyProtectedMember
from torch.optim.lr_scheduler import _LRScheduler

from core.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from core.algorithms.onpolicy_sync.policy import ActorCriticModel
from core.algorithms.onpolicy_sync.storage import RolloutStorage
from core.algorithms.onpolicy_sync.vector_sampled_tasks import (
    VectorSampledTasks,
    COMPLETE_TASK_METRICS_KEY,
)
from core.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from core.base_abstractions.misc import RLStepResult
from utils.experiment_utils import (
    set_deterministic_cudnn,
    set_seed,
    Builder,
    TrainingPipeline,
    PipelineStage,
    LoggingPackage,
)
from utils.system import get_logger
from utils.tensor_utils import (
    batch_observations,
    to_device_recursively,
    detach_recursively,
)
from utils.viz_utils import VizSuite
from utils import spaces_utils as su


class OnPolicyRLEngine(object):
    """The reinforcement learning primary controller.

    This `OnPolicyRLEngine` class handles all training, validation, and
    testing as well as logging and checkpointing. You are not expected
    to instantiate this class yourself, instead you should define an
    experiment which will then be used to instantiate an
    `OnPolicyRLEngine` and perform any desired tasks.
    """

    def __init__(
        self,
        experiment_name: str,
        config: ExperimentConfig,
        results_queue: mp.Queue,  # to output aggregated results
        checkpoints_queue: Optional[
            mp.Queue
        ],  # to write/read (trainer/evaluator) ready checkpoints
        checkpoints_dir: str,
        mode: str = "train",
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        mp_ctx: Optional[BaseContext] = None,
        worker_id: int = 0,
        num_workers: int = 1,
        device: Union[str, torch.device, int] = "cpu",
        distributed_port: int = 0,
        deterministic_agents: bool = False,
        max_sampler_processes_per_worker: Optional[int] = None,
        **kwargs,
    ):
        """Initializer.

        # Parameters

        config : The ExperimentConfig defining the experiment to run.
        output_dir : Root directory at which checkpoints and logs should be saved.
        seed : Seed used to encourage deterministic behavior (it is difficult to ensure
            completely deterministic behavior due to CUDA issues and nondeterminism
            in environments).
        mode : "train", "valid", or "test".
        deterministic_cudnn : Whether or not to use deterministic cudnn. If `True` this may lower
            training performance this is necessary (but not sufficient) if you desire
            deterministic behavior.
        extra_tag : An additional label to add to the experiment when saving tensorboard logs.
        """
        self.config = config
        self.results_queue = results_queue
        self.checkpoints_queue = checkpoints_queue
        self.mp_ctx = mp_ctx
        self.checkpoints_dir = checkpoints_dir
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.device = torch.device("cpu") if device == -1 else torch.device(device)  # type: ignore
        self.distributed_port = distributed_port

        self.mode = mode.lower()
        assert self.mode in [
            "train",
            "valid",
            "test",
        ], "Only train, valid, test modes supported"

        self.deterministic_cudnn = deterministic_cudnn
        if self.deterministic_cudnn:
            set_deterministic_cudnn()

        self.seed = seed
        set_seed(self.seed)

        self.experiment_name = experiment_name

        assert (
            max_sampler_processes_per_worker is None
            or max_sampler_processes_per_worker >= 1
        ), "`max_sampler_processes_per_worker` must be either `None` or a positive integer."
        self.max_sampler_processes_per_worker = max_sampler_processes_per_worker

        machine_params = config.machine_params(self.mode)
        self.machine_params: MachineParams
        if isinstance(machine_params, MachineParams):
            self.machine_params = machine_params
        else:
            self.machine_params = MachineParams(**machine_params)

        self.num_samplers_per_worker = self.machine_params.nprocesses
        self.num_samplers = self.num_samplers_per_worker[self.worker_id]

        self._vector_tasks: Optional[VectorSampledTasks] = None

        self.sensor_preprocessor_graph = None
        self.actor_critic: Optional[ActorCriticModel] = None
        if self.num_samplers > 0:
            if self.machine_params.sensor_preprocessor_graph is not None:
                # centralized observation set,
                self.sensor_preprocessor_graph = self.machine_params.sensor_preprocessor_graph.to(
                    self.device
                )
                set_seed(self.seed)
                self.actor_critic = cast(
                    ActorCriticModel,
                    self.config.create_model(
                        sensor_preprocessor_graph=self.sensor_preprocessor_graph
                    ),
                ).to(self.device)
            else:
                # no observation set
                set_seed(self.seed)
                self.actor_critic = cast(
                    ActorCriticModel, self.config.create_model()
                ).to(self.device)

        self.is_distributed = False
        self.store: Optional[torch.distributed.TCPStore] = None  # type:ignore
        if self.num_workers > 1:
            self.store = torch.distributed.TCPStore(  # type:ignore
                "127.0.0.1",
                self.distributed_port,
                self.num_workers,
                self.worker_id == 0,
            )
            cpu_device = self.device == torch.device("cpu")  # type:ignore

            dist.init_process_group(  # type:ignore
                backend="gloo" if cpu_device else "nccl",
                store=self.store,
                rank=self.worker_id,
                world_size=self.num_workers,
            )
            self.is_distributed = True

        self.deterministic_agents = deterministic_agents

        self._is_closed: bool = False

        self.training_pipeline: Optional[TrainingPipeline] = None

        # Keeping track of metrics during training/inference
        self.single_process_metrics_queue: queue.Queue = queue.Queue()

    @property
    def vector_tasks(self) -> VectorSampledTasks:
        if self._vector_tasks is None and self.num_samplers > 0:
            if self.is_distributed:
                total_processes = sum(
                    self.num_samplers_per_worker
                )  # TODO this will break the fixed seed for multi-device test
            else:
                total_processes = self.num_samplers

            seeds = self.worker_seeds(
                total_processes,
                initial_seed=self.seed,  # do not update the RNG state (creation might happen after seed resetting)
            )

            self._vector_tasks = VectorSampledTasks(
                make_sampler_fn=self.config.make_sampler_fn,
                sampler_fn_args=self.get_sampler_fn_args(seeds),
                multiprocessing_start_method="forkserver"
                if self.mp_ctx is None
                else None,
                mp_ctx=self.mp_ctx,
                max_processes=self.max_sampler_processes_per_worker,
            )
        return self._vector_tasks

    @staticmethod
    def worker_seeds(nprocesses: int, initial_seed: Optional[int]) -> List[int]:
        """Create a collection of seeds for workers without modifying the RNG
        state."""
        rstate = None  # type:ignore
        if initial_seed is not None:
            rstate = random.getstate()
            random.seed(initial_seed)
        seeds = [random.randint(0, (2 ** 31) - 1) for _ in range(nprocesses)]
        if initial_seed is not None:
            random.setstate(rstate)
        return seeds

    def get_sampler_fn_args(self, seeds: Optional[List[int]] = None):
        sampler_devices = self.machine_params.sampler_devices

        if self.mode == "train":
            fn = self.config.train_task_sampler_args
        elif self.mode == "valid":
            fn = self.config.valid_task_sampler_args
        elif self.mode == "test":
            fn = self.config.test_task_sampler_args
        else:
            raise NotImplementedError(
                "self.mode must be one of `train`, `valid` or `test`."
            )

        if self.is_distributed:
            total_processes = sum(self.num_samplers_per_worker)
            process_offset = sum(self.num_samplers_per_worker[: self.worker_id])
        else:
            total_processes = self.num_samplers
            process_offset = 0

        sampler_devices_as_ints: Optional[List[int]] = None
        if (
            self.is_distributed or self.mode == "test"
        ) and self.device.index is not None:
            sampler_devices_as_ints = [self.device.index]
        elif sampler_devices is not None:
            sampler_devices_as_ints = [
                -1 if sd.index is None else sd.index for sd in sampler_devices
            ]

        return [
            fn(
                process_ind=process_offset + it,
                total_processes=total_processes,
                devices=sampler_devices_as_ints,
                seeds=seeds,
            )
            for it in range(self.num_samplers)
        ]

    def checkpoint_load(
        self, ckpt: Union[str, Dict[str, Any]]
    ) -> Dict[str, Union[Dict[str, Any], torch.Tensor, float, int, str, List]]:
        if isinstance(ckpt, str):
            get_logger().info(
                "{} worker {} loading checkpoint from {}".format(
                    self.mode, self.worker_id, ckpt
                )
            )
            # Map location CPU is almost always better than mapping to a CUDA device.
            ckpt = torch.load(ckpt, map_location="cpu")

        ckpt = cast(
            Dict[str, Union[Dict[str, Any], torch.Tensor, float, int, str, List]], ckpt,
        )

        self.actor_critic.load_state_dict(ckpt["model_state_dict"])  # type:ignore

        return ckpt

    # aggregates task metrics currently in queue
    def aggregate_task_metrics(
        self, logging_pkg: LoggingPackage, num_tasks: int = -1,
    ) -> LoggingPackage:
        done = num_tasks == 0
        num_empty_tasks_dequeued = 0
        while not done:
            try:
                # This queue is on this process so we should be able to let the timeout be small
                # TODO: This should be refactored so that single_process_metrics_queue is a list
                metrics_dict = self.single_process_metrics_queue.get(timeout=0.01)

                num_empty_tasks_dequeued += not logging_pkg.add_metrics_dict(
                    single_task_metrics_dict=metrics_dict
                )

                if num_tasks > 0:
                    num_tasks -= 1
                done = num_tasks == 0

            except queue.Empty:
                if num_tasks <= 0:
                    break
                else:
                    get_logger().error(
                        "Metrics out queue is empty after a short second wait."
                        " This should only happen if a positive number of `num_tasks` were"
                        " set during testing but the queue did not contain this number of entries."
                        " Please file an issue at https://github.com/allenai/allenact/issues."
                    )
                    break

        if num_empty_tasks_dequeued != 0:
            get_logger().warning(
                "Discarded {} empty task metrics".format(num_empty_tasks_dequeued)
            )

        return logging_pkg

    def _preprocess_observations(self, batched_observations):
        if self.sensor_preprocessor_graph is None:
            return batched_observations
        return self.sensor_preprocessor_graph.get_observations(batched_observations)

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

        # Group samplers along new dim:
        batch = batch_observations(running, device=self.device)

        return len(paused), keep, batch

    def initialize_rollouts(self, rollouts, visualizer: Optional[VizSuite] = None):
        observations = self.vector_tasks.get_observations()

        npaused, keep, batch = self.remove_paused(observations)
        if npaused > 0:
            rollouts.sampler_select(keep)
        rollouts.to(self.device)
        rollouts.insert_observations(
            self._preprocess_observations(batch) if len(keep) > 0 else batch
        )
        if visualizer is not None and len(keep) > 0:
            visualizer.collect(vector_task=self.vector_tasks, alive=keep)
        return npaused

    def act(self, rollouts: RolloutStorage):
        with torch.no_grad():
            step_observation = rollouts.pick_observation_step(rollouts.step)
            memory = rollouts.pick_memory_step(rollouts.step)
            prev_actions = rollouts.pick_prev_actions_step(rollouts.step)
            actor_critic_output, memory = self.actor_critic(
                step_observation,
                memory,
                prev_actions,
                rollouts.masks[rollouts.step : rollouts.step + 1],
            )

        # Assume actions do not contain a step dimension
        actions = (
            actor_critic_output.distributions.sample()
            if not self.deterministic_agents
            else actor_critic_output.distributions.mode()
        )

        return actions, actor_critic_output, memory, step_observation

    @staticmethod
    def _active_memory(memory, keep):
        return memory.sampler_select(keep) if memory is not None else memory

    def probe(self, dones: List[bool], npaused, period=100000):
        """
        Debugging util. When called from self.collect_rollout_step(...), calls render for the 0-th task sampler of the
        0-th distributed worker for the first beginning episode spaced at least period steps from the beginning
        of the previous one.

        For valid, train, it currently renders all episodes for the 0-th task sampler of the
        0-th distributed worker. If this is not wanted, it must be hard-coded for now below.

        :param dones: dones list from self.collect_rollout_step(...)
        :param npaused: number of newly paused tasks returned by self.removed_paused(...)
        :param period: minimal spacing in sampled steps between the beginning of episodes to be shown.
        """
        sampler_id = 0
        done = dones[sampler_id]
        if self.mode != "train":
            setattr(
                self, "_probe_npaused", getattr(self, "_probe_npaused", 0) + npaused
            )
            if self._probe_npaused == self.num_samplers:
                del self._probe_npaused
                return
            period = 0
        if self.worker_id == 0:
            if done:
                if period > 0 and (
                    getattr(self, "_probe_steps", None) is None
                    or (
                        self._probe_steps < 0
                        and (self.training_pipeline.total_steps + self._probe_steps)
                        >= period
                    )
                ):
                    self._probe_steps = self.training_pipeline.total_steps
            if period == 0 or (
                getattr(self, "_probe_steps", None) is not None
                and self._probe_steps >= 0
                and ((self.training_pipeline.total_steps - self._probe_steps) < period)
            ):
                if (
                    period == 0
                    or not done
                    or self._probe_steps == self.training_pipeline.total_steps
                ):
                    self.vector_tasks.call_at(sampler_id, "render", ["human"])
                else:
                    self._probe_steps = -self._probe_steps

    def collect_rollout_step(self, rollouts: RolloutStorage, visualizer=None) -> int:
        actions, actor_critic_output, memory, _ = self.act(rollouts=rollouts)

        # Flatten actions
        flat_actions = su.flatten(self.actor_critic.action_space, actions)

        assert len(flat_actions.shape) == 3, (
            "Distribution samples must include step and task sampler dimensions [step, sampler, ...]. The simplest way"
            "to accomplish this is to pass param tensors (like `logits` in a `CategoricalDistr`) with these dimensions"
            "to the Distribution."
        )

        # Convert flattened actions into list of actions and send them
        outputs: List[RLStepResult] = self.vector_tasks.step(
            su.action_list(self.actor_critic.action_space, flat_actions)
        )

        # Save after task completion metrics
        for step_result in outputs:
            if COMPLETE_TASK_METRICS_KEY in step_result.info:
                self.single_process_metrics_queue.put(
                    step_result.info[COMPLETE_TASK_METRICS_KEY]
                )
                del step_result.info[COMPLETE_TASK_METRICS_KEY]

        rewards: Union[List, torch.Tensor]
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=self.device,  # type:ignore
        )

        # We want rewards to have dimensions [sampler, reward]
        if len(rewards.shape) == 1:
            # Rewards are of shape [sampler,]
            rewards = rewards.unsqueeze(-1)
        elif len(rewards.shape) > 1:
            raise NotImplementedError()

        # If done then clean the history of observations.
        masks = torch.tensor(
            [0.0 if done else 1.0 for done in dones],
            dtype=torch.float32,
            device=self.device,  # type:ignore
        ).view(
            -1, 1
        )  # [sampler, 1]

        npaused, keep, batch = self.remove_paused(observations)

        # TODO self.probe(...) can be useful for debugging (we might want to control it from main?)
        # self.probe(dones, npaused)

        if npaused > 0:
            rollouts.sampler_select(keep)

        rollouts.insert(
            observations=self._preprocess_observations(batch)
            if len(keep) > 0
            else batch,
            memory=self._active_memory(memory, keep),
            actions=flat_actions[0, keep],
            action_log_probs=actor_critic_output.distributions.log_prob(actions)[
                0, keep
            ],
            value_preds=actor_critic_output.values[0, keep],
            rewards=rewards[keep],
            masks=masks[keep],
        )

        # TODO we always miss tensors for the last action in the last episode of each worker
        if visualizer is not None:
            if len(keep) > 0:
                visualizer.collect(
                    rollout=rollouts,
                    vector_task=self.vector_tasks,
                    alive=keep,
                    actor_critic=actor_critic_output,
                )
            else:
                visualizer.collect(actor_critic=actor_critic_output)

        return npaused

    def close(self, verbose=True):
        if "_is_closed" in self.__dict__ and self._is_closed:
            return

        def logif(s: Union[str, Exception]):
            if verbose:
                if isinstance(s, str):
                    get_logger().info(s)
                elif isinstance(s, Exception):
                    get_logger().error(traceback.format_exc())
                else:
                    raise NotImplementedError()

        if "_vector_tasks" in self.__dict__ and self._vector_tasks is not None:
            try:
                logif(
                    "{} worker {} Closing OnPolicyRLEngine.vector_tasks.".format(
                        self.mode, self.worker_id
                    )
                )
                self._vector_tasks.close()
                logif("{} worker {} Closed.".format(self.mode, self.worker_id))
            except Exception as e:
                logif(
                    "{} worker {} Exception raised when closing OnPolicyRLEngine.vector_tasks:".format(
                        self.mode, self.worker_id
                    )
                )
                logif(e)

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
        experiment_name: str,
        config: ExperimentConfig,
        results_queue: mp.Queue,
        checkpoints_queue: Optional[mp.Queue],
        checkpoints_dir: str = "",
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        mp_ctx: Optional[BaseContext] = None,
        worker_id: int = 0,
        num_workers: int = 1,
        device: Union[str, torch.device, int] = "cpu",
        distributed_port: int = 0,
        deterministic_agents: bool = False,
        distributed_preemption_threshold: float = 0.7,
        max_sampler_processes_per_worker: Optional[int] = None,
        **kwargs,
    ):
        kwargs["mode"] = "train"
        super().__init__(
            experiment_name=experiment_name,
            config=config,
            results_queue=results_queue,
            checkpoints_queue=checkpoints_queue,
            checkpoints_dir=checkpoints_dir,
            seed=seed,
            deterministic_cudnn=deterministic_cudnn,
            mp_ctx=mp_ctx,
            worker_id=worker_id,
            num_workers=num_workers,
            device=device,
            distributed_port=distributed_port,
            deterministic_agents=deterministic_agents,
            max_sampler_processes_per_worker=max_sampler_processes_per_worker,
            **kwargs,
        )

        self.actor_critic.train()

        self.training_pipeline: TrainingPipeline = config.training_pipeline()

        self.optimizer: optim.optimizer.Optimizer = self.training_pipeline.optimizer_builder(
            params=[p for p in self.actor_critic.parameters() if p.requires_grad]
        )

        # noinspection PyProtectedMember
        self.lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        if self.training_pipeline.lr_scheduler_builder is not None:
            self.lr_scheduler = self.training_pipeline.lr_scheduler_builder(
                optimizer=self.optimizer
            )

        if self.is_distributed:
            # Tracks how many workers have finished their rollout
            self.num_workers_done = torch.distributed.PrefixStore(  # type:ignore
                "num_workers_done", self.store
            )
            # Tracks the number of steps taken by each worker in current rollout
            self.num_workers_steps = torch.distributed.PrefixStore(  # type:ignore
                "num_workers_steps", self.store
            )
            self.distributed_preemption_threshold = distributed_preemption_threshold
            # Flag for finished worker in current epoch
            self.offpolicy_epoch_done = torch.distributed.PrefixStore(  # type:ignore
                "offpolicy_epoch_done", self.store
            )
        else:
            self.num_workers_done = None
            self.num_workers_steps = None
            self.distributed_preemption_threshold = 1.0
            self.offpolicy_epoch_done = None

        # Keeping track of training state
        self.tracking_info: Dict[str, List] = defaultdict(lambda: [])
        self.former_steps: Optional[int] = None
        self.last_log: Optional[int] = None
        self.last_save: Optional[int] = None

    def advance_seed(
        self, seed: Optional[int], return_same_seed_per_worker=False
    ) -> Optional[int]:
        if seed is None:
            return seed
        seed = (seed ^ (self.training_pipeline.total_steps + 1)) % (
            2 ** 31 - 1
        )  # same seed for all workers

        if (not return_same_seed_per_worker) and (
            self.mode == "train" or self.mode == "test"
        ):
            return self.worker_seeds(self.num_workers, seed)[
                self.worker_id
            ]  # doesn't modify the current rng state
        else:
            return self.worker_seeds(1, seed)[0]  # doesn't modify the current rng state

    def deterministic_seeds(self) -> None:
        if self.seed is not None:
            set_seed(self.advance_seed(self.seed))  # known state for all workers
            seeds = self.worker_seeds(
                self.num_samplers, None
            )  # use latest seed for workers and update rng state
            self.vector_tasks.set_seeds(seeds)

    def checkpoint_save(self) -> str:
        model_path = os.path.join(
            self.checkpoints_dir,
            "exp_{}__stage_{:02d}__steps_{:012d}.pt".format(
                self.experiment_name,
                self.training_pipeline.current_stage_index,
                self.training_pipeline.total_steps,
            ),
        )

        save_dict = {
            "model_state_dict": self.actor_critic.state_dict(),  # type:ignore
            "total_steps": self.training_pipeline.total_steps,  # Total steps including current stage
            "optimizer_state_dict": self.optimizer.state_dict(),  # type: ignore
            "training_pipeline_state_dict": self.training_pipeline.state_dict(),
            "trainer_seed": self.seed,
        }

        if self.lr_scheduler is not None:
            save_dict["scheduler_state"] = cast(
                _LRScheduler, self.lr_scheduler
            ).state_dict()

        torch.save(save_dict, model_path)
        return model_path

    def checkpoint_load(
        self, ckpt: Union[str, Dict[str, Any]], restart_pipeline: bool = False
    ) -> Dict[str, Union[Dict[str, Any], torch.Tensor, float, int, str, List]]:
        ckpt = super().checkpoint_load(ckpt)

        self.training_pipeline.load_state_dict(
            cast(Dict[str, Any], ckpt["training_pipeline_state_dict"])
        )
        if restart_pipeline:
            self.training_pipeline.restart_pipeline()
        else:
            self.seed = cast(int, ckpt["trainer_seed"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # type: ignore
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(ckpt["scheduler_state"])  # type: ignore

        self.deterministic_seeds()

        return ckpt

    def _get_loss(self, loss_name) -> AbstractActorCriticLoss:
        assert (
            loss_name in self.training_pipeline.named_losses
        ), "undefined referenced loss {}".format(loss_name)
        if isinstance(self.training_pipeline.named_losses[loss_name], Builder):
            return cast(
                Builder[AbstractActorCriticLoss],
                self.training_pipeline.named_losses[loss_name],
            )()
        else:
            return cast(
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

        if (
            hasattr(self.machine_params, field)
            and getattr(self.machine_params, field) is not None
        ):
            return getattr(self.machine_params, field)

        if allow_none:
            return None
        else:
            raise RuntimeError("missing value for {}".format(field))

    @property
    def step_count(self):
        return self.training_pipeline.current_stage.steps_taken_in_stage

    @step_count.setter
    def step_count(self, val: int):
        self.training_pipeline.current_stage.steps_taken_in_stage = val

    @property
    def log_interval(self):
        return self.training_pipeline.metric_accumulate_interval

    def act(self, rollouts: RolloutStorage):
        actions, actor_critic_output, memory, step_observation = super().act(
            rollouts=rollouts
        )

        if self.is_distributed:
            # TODO this is inaccurate/hacky, but gets synchronized after each rollout
            approx_steps = (
                self.step_count - self.former_steps
            ) * self.num_workers + self.former_steps
        else:
            approx_steps = self.step_count  # this is actually accurate

        num_active_samplers = actor_critic_output.values.shape[1]

        if self.training_pipeline.current_stage.teacher_forcing is not None:
            if self.training_pipeline.current_stage.teacher_forcing(approx_steps) > 0:
                actions, enforce_info = self.apply_teacher_forcing(
                    actions, step_observation, approx_steps
                )
                empirical_enforced = enforce_info["teacher_forcing_mask"].mean().item()
            else:
                empirical_enforced = 0

            teacher_force_info = {
                "teacher_ratio/sampled": empirical_enforced,
                "teacher_ratio/enforced": self.training_pipeline.current_stage.teacher_forcing(
                    approx_steps
                ),
            }
            self.tracking_info["teacher"].append(
                ("teacher_package", teacher_force_info, actions.nelement())
            )

        self.step_count += num_active_samplers

        return actions, actor_critic_output, memory, step_observation

    def update(self, rollouts: RolloutStorage):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        for e in range(self.training_pipeline.update_repeats):
            data_generator = rollouts.recurrent_generator(
                advantages, self.training_pipeline.num_mini_batch
            )

            for bit, batch in enumerate(data_generator):
                # masks is always [steps, samplers, 1]:
                num_rollout_steps, num_samplers = batch["masks"].shape[:2]
                bsize = num_rollout_steps * num_samplers

                actor_critic_output, memory = self.actor_critic(
                    observations=batch["observations"],
                    memory=batch["memory"],
                    prev_actions=batch["prev_actions"],
                    masks=batch["masks"],
                )

                info: Dict[str, float] = {
                    "lr": self.optimizer.param_groups[0]["lr"]  # type: ignore
                }

                total_loss: Optional[torch.Tensor] = None
                for loss_name in self.training_pipeline.current_stage_losses:
                    loss, loss_weight = (
                        self.training_pipeline.current_stage_losses[loss_name],
                        self.training_pipeline.current_stage_loss_weights[loss_name],
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

                    for key in current_info:
                        info[loss_name + "/" + key] = current_info[key]

                assert (
                    total_loss is not None
                ), "No losses specified for training in stage {}".format(
                    self.training_pipeline.current_stage_index
                )

                info["total_loss"] = total_loss.item()
                self.tracking_info["update"].append(("update_package", info, bsize))

                self.backprop_step(total_loss)

        # # TODO Unit test to ensure correctness of distributed infrastructure
        # state_dict = self.actor_critic.state_dict()
        # keys = sorted(list(state_dict.keys()))
        # get_logger().debug(
        #     "worker {} param 0 {} param -1 {}".format(
        #         self.worker_id,
        #         state_dict[keys[0]].flatten()[0],
        #         state_dict[keys[-1]].flatten()[-1],
        #     )
        # )

    def make_offpolicy_iterator(
        self, data_iterator_builder: Callable[..., Iterator],
    ):
        stage = self.training_pipeline.current_stage

        if self.num_workers == 1:
            rollouts_per_worker: Sequence[int] = [self.num_samplers]
        else:
            rollouts_per_worker = self.num_samplers_per_worker

        # common seed for all workers (in case we wish to shuffle the full dataset before iterating on one partition)
        seed = self.advance_seed(self.seed, return_same_seed_per_worker=True)

        kwargs = stage.offpolicy_component.data_iterator_kwargs_generator(
            self.worker_id, rollouts_per_worker, seed
        )

        offpolicy_iterator = data_iterator_builder(**kwargs)

        stage.offpolicy_memory.clear()
        if stage.offpolicy_epochs is None:
            stage.offpolicy_epochs = 0
        else:
            stage.offpolicy_epochs += 1

        if self.is_distributed:
            self.offpolicy_epoch_done.set("offpolicy_epoch_done", str(0))
            dist.barrier()  # sync

        return offpolicy_iterator

    def backprop_step(self, total_loss):
        self.optimizer.zero_grad()  # type: ignore
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()

        if self.is_distributed:
            # From https://github.com/pytorch/pytorch/issues/43135
            reductions = []
            for p in self.actor_critic.parameters():
                # you can also organize grads to larger buckets to make allreduce more efficient
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    reductions.append(
                        dist.all_reduce(p.grad, async_op=True,)
                    )  # synchronize
            for reduction in reductions:
                reduction.wait()

        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.training_pipeline.max_grad_norm,  # type: ignore
        )
        self.optimizer.step()  # type: ignore

    def offpolicy_update(
        self,
        updates: int,
        data_iterator: Optional[Iterator],
        data_iterator_builder: Callable[..., Iterator],
    ) -> Iterator:
        stage = self.training_pipeline.current_stage

        current_steps = 0
        if self.is_distributed:
            self.num_workers_steps.set("steps", str(0))
            dist.barrier()

        for e in range(updates):
            if data_iterator is None:
                data_iterator = self.make_offpolicy_iterator(data_iterator_builder)

            try:
                batch = next(data_iterator)
            except StopIteration:
                batch = None
                if self.is_distributed:
                    self.offpolicy_epoch_done.add("offpolicy_epoch_done", 1)

            if self.is_distributed:
                dist.barrier()  # sync after every batch!
                if int(self.offpolicy_epoch_done.get("offpolicy_epoch_done")) != 0:
                    batch = None

            if batch is None:
                data_iterator = self.make_offpolicy_iterator(data_iterator_builder)
                # TODO: (batch, bsize) from iterator instead of waiting for the loss?
                batch = next(data_iterator)

            batch = to_device_recursively(batch, device=self.device, inplace=True)

            info: Dict[str, float] = dict()
            info["lr"] = self.optimizer.param_groups[0]["lr"]  # type: ignore

            bsize: Optional[int] = None

            total_loss: Optional[torch.Tensor] = None
            for loss_name in stage.offpolicy_named_loss_weights:
                loss, loss_weight = (
                    self.training_pipeline.current_stage_offpolicy_losses[loss_name],
                    stage.offpolicy_named_loss_weights[loss_name],
                )

                current_loss, current_info, stage.offpolicy_memory, bsize = loss.loss(
                    model=self.actor_critic,
                    batch=batch,
                    step_count=self.step_count,
                    memory=stage.offpolicy_memory,
                )
                if total_loss is None:
                    total_loss = loss_weight * current_loss
                else:
                    total_loss = total_loss + loss_weight * current_loss

                for key in current_info:
                    info["offpolicy/" + loss_name + "/" + key] = current_info[key]

            assert (
                total_loss is not None
            ), "No offline losses specified for training in stage {}".format(
                self.training_pipeline.current_stage_index
            )

            info["offpolicy/total_loss"] = total_loss.item()
            info["offpolicy/epoch"] = stage.offpolicy_epochs
            self.tracking_info["offpolicy_update"].append(
                ("offpolicy_update_package", info, bsize)
            )

            self.backprop_step(total_loss)

            stage.offpolicy_memory = detach_recursively(
                input=stage.offpolicy_memory, inplace=True
            )

            if self.is_distributed:
                self.num_workers_steps.add("steps", bsize)  # counts samplers x steps
            else:
                current_steps += bsize

        if self.is_distributed:
            dist.barrier()
            stage.offpolicy_steps_taken_in_stage += int(
                self.num_workers_steps.get("steps")
            )
            dist.barrier()
        else:
            stage.offpolicy_steps_taken_in_stage += current_steps

        return data_iterator

    def apply_teacher_forcing(
        self, actions: Any, step_observation: Dict[str, torch.Tensor], step_count: int,
    ):
        # Recall that the last dimension of the expert action tensor has its last element equal to
        # 0 if the expert action could not be computed and otherwise equals 1.
        tf_mask_shape = step_observation["expert_action"].shape[:-1] + (1,)
        expert_actions = step_observation["expert_action"][..., :-1]
        expert_action_exists_mask = step_observation["expert_action"][..., -1:]

        actions = su.flatten(self.actor_critic.action_space, actions)

        assert (
            expert_actions.shape == actions.shape
        ), "expert actions shape {} doesn't match the model's {}".format(
            expert_actions.shape, actions.shape
        )

        teacher_forcing_mask = (
            torch.distributions.bernoulli.Bernoulli(
                torch.tensor(
                    self.training_pipeline.current_stage.teacher_forcing(step_count)
                )
            )
            .sample(tf_mask_shape)
            .long()
            .to(self.device)
        ) * expert_action_exists_mask

        extended_shape = teacher_forcing_mask.shape + (1,) * (
            len(actions.shape) - len(teacher_forcing_mask.shape)
        )

        actions = torch.where(
            teacher_forcing_mask.byte().view(extended_shape), expert_actions, actions
        )

        return (
            su.unflatten(self.actor_critic.action_space, actions),
            {"teacher_forcing_mask": teacher_forcing_mask},
        )

    def send_package(self, tracking_info: Dict[str, List]):
        logging_pkg = LoggingPackage(
            mode=self.mode,
            training_steps=self.training_pipeline.total_steps,
            off_policy_steps=self.training_pipeline.total_offpolicy_steps,
            pipeline_stage=self.training_pipeline.current_stage_index,
        )

        self.aggregate_task_metrics(logging_pkg=logging_pkg)

        for (info_type, train_info_dict, n) in itertools.chain(*tracking_info.values()):
            if n < 0:
                get_logger().warning(
                    f"Obtained a train_info_dict with {n} elements."
                    f" Full info: ({info_type}, {train_info_dict}, {n})."
                )
            else:
                logging_pkg.add_train_info_dict(train_info_dict=train_info_dict, n=n)

        self.results_queue.put(logging_pkg)

    def run_pipeline(self, rollouts: RolloutStorage):
        self.initialize_rollouts(rollouts)
        self.tracking_info.clear()

        self.last_log = self.training_pipeline.total_steps
        self.last_save = self.training_pipeline.total_steps

        offpolicy_data_iterator: Optional[Iterator] = None

        while True:
            self.training_pipeline.before_rollout()
            if self.training_pipeline.current_stage is None:
                break

            if self.is_distributed:
                self.num_workers_done.set("done", str(0))
                self.num_workers_steps.set("steps", str(0))
                # Ensure all workers are done before incrementing num_workers_{steps, done}
                dist.barrier()

            self.former_steps = self.step_count
            for step in range(self.training_pipeline.num_steps):
                self.collect_rollout_step(rollouts=rollouts)
                if self.is_distributed:
                    # Preempt stragglers
                    # Each worker will stop collecting steps for the current rollout whenever a
                    # 100 * distributed_preemption_threshold percentage of workers are finished collecting their
                    # rollout steps and we have collected at least 25% but less than 90% of the steps.
                    num_done = int(self.num_workers_done.get("done"))
                    if (
                        num_done
                        > self.distributed_preemption_threshold * self.num_workers
                        and 0.25 * self.training_pipeline.num_steps
                        <= step
                        < 0.9 * self.training_pipeline.num_steps
                    ):
                        get_logger().debug(
                            "{} worker {} narrowed rollouts after {} steps (out of {}) with {} workers done".format(
                                self.mode, self.worker_id, rollouts.step, step, num_done
                            )
                        )
                        rollouts.narrow()
                        break

            with torch.no_grad():
                actor_critic_output, _ = self.actor_critic(
                    observations=rollouts.pick_observation_step(-1),
                    memory=rollouts.pick_memory_step(-1),
                    prev_actions=rollouts.prev_actions[-1:],
                    masks=rollouts.masks[-1:],
                )

            if self.is_distributed:
                # Mark that a worker is done collecting experience
                self.num_workers_done.add("done", 1)
                self.num_workers_steps.add("steps", self.step_count - self.former_steps)

                # Ensure all workers are done before updating step counter
                dist.barrier()

                ndone = int(self.num_workers_done.get("done"))
                assert (
                    ndone == self.num_workers
                ), "# workers done {} != # workers {}".format(ndone, self.num_workers)

                # get the actual step_count
                self.step_count = (
                    int(self.num_workers_steps.get("steps")) + self.former_steps
                )

            rollouts.compute_returns(
                next_value=actor_critic_output.values.detach(),
                use_gae=self.training_pipeline.use_gae,
                gamma=self.training_pipeline.gamma,
                tau=self.training_pipeline.gae_lambda,
            )

            self.update(rollouts=rollouts)  # here we synchronize
            self.training_pipeline.rollout_count += 1

            rollouts.after_update()

            if self.training_pipeline.current_stage.offpolicy_component is not None:
                offpolicy_component = (
                    self.training_pipeline.current_stage.offpolicy_component
                )
                offpolicy_data_iterator = self.offpolicy_update(
                    updates=offpolicy_component.updates,
                    data_iterator=offpolicy_data_iterator,
                    data_iterator_builder=offpolicy_component.data_iterator_builder,
                )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch=self.training_pipeline.total_steps)

            if (
                self.training_pipeline.total_steps - self.last_log >= self.log_interval
                or self.training_pipeline.current_stage.is_complete
            ):
                self.send_package(tracking_info=self.tracking_info)
                self.tracking_info.clear()
                self.last_log = self.training_pipeline.total_steps

            # save for every interval-th episode or for the last epoch
            if (
                self.checkpoints_dir != ""
                and self.training_pipeline.save_interval > 0
                and (
                    self.training_pipeline.total_steps - self.last_save
                    >= self.training_pipeline.save_interval
                    or self.training_pipeline.current_stage.is_complete
                )
            ):
                self.deterministic_seeds()
                if self.worker_id == 0:
                    model_path = self.checkpoint_save()
                    if self.checkpoints_queue is not None:
                        self.checkpoints_queue.put(("eval", model_path))
                self.last_save = self.training_pipeline.total_steps

            if (self.training_pipeline.advance_scene_rollout_period is not None) and (
                self.training_pipeline.rollout_count
                % self.training_pipeline.advance_scene_rollout_period
                == 0
            ):
                get_logger().info(
                    "{} worker {} Force advance tasks with {} rollouts".format(
                        self.mode, self.worker_id, self.training_pipeline.rollout_count
                    )
                )
                self.vector_tasks.next_task(force_advance_scene=True)
                self.initialize_rollouts(rollouts)

    def train(
        self, checkpoint_file_name: Optional[str] = None, restart_pipeline: bool = False
    ):
        assert self.mode == "train", "train only to be called from a train instance"

        training_completed_successfully = False
        try:
            if checkpoint_file_name is not None:
                self.checkpoint_load(checkpoint_file_name, restart_pipeline)

            self.run_pipeline(
                RolloutStorage(
                    num_steps=self.training_pipeline.num_steps,
                    num_samplers=self.num_samplers,
                    actor_critic=self.actor_critic
                    if isinstance(self.actor_critic, ActorCriticModel)
                    else cast(ActorCriticModel, self.actor_critic.module),
                )
            )

            training_completed_successfully = True
        except KeyboardInterrupt:
            get_logger().info(
                "KeyboardInterrupt. Terminating {} worker {}".format(
                    self.mode, self.worker_id
                )
            )
        except Exception:
            get_logger().error(
                "Encountered Exception. Terminating {} worker {}".format(
                    self.mode, self.worker_id
                )
            )
            get_logger().exception(traceback.format_exc())
        finally:
            if training_completed_successfully:
                if self.worker_id == 0:
                    self.results_queue.put(("train_stopped", 0))
                get_logger().info(
                    "{} worker {} COMPLETE".format(self.mode, self.worker_id)
                )
            else:
                self.results_queue.put(("train_stopped", 1 + self.worker_id))
            self.close()


class OnPolicyInference(OnPolicyRLEngine):
    def __init__(
        self,
        config: ExperimentConfig,
        results_queue: mp.Queue,  # to output aggregated results
        checkpoints_queue: mp.Queue,  # to write/read (trainer/evaluator) ready checkpoints
        checkpoints_dir: str = "",
        mode: str = "valid",  # or "test"
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        mp_ctx: Optional[BaseContext] = None,
        device: Union[str, torch.device, int] = "cpu",
        deterministic_agents: bool = False,
        worker_id: int = 0,
        num_workers: int = 1,
        distributed_port: int = 0,
        **kwargs,
    ):
        super().__init__(
            experiment_name="",
            config=config,
            results_queue=results_queue,
            checkpoints_queue=checkpoints_queue,
            checkpoints_dir=checkpoints_dir,
            mode=mode,
            seed=seed,
            deterministic_cudnn=deterministic_cudnn,
            mp_ctx=mp_ctx,
            deterministic_agents=deterministic_agents,
            device=device,
            worker_id=worker_id,
            num_workers=num_workers,
            distributed_port=distributed_port,
            **kwargs,
        )

    def run_eval(
        self,
        checkpoint_file_name: str,
        rollout_steps=100,
        visualizer: Optional[VizSuite] = None,
        update_secs=20,
    ) -> LoggingPackage:
        assert self.actor_critic is not None, "called run_eval with no actor_critic"

        ckpt = self.checkpoint_load(checkpoint_file_name)
        total_steps = cast(int, ckpt["total_steps"])

        rollouts = RolloutStorage(
            num_steps=rollout_steps,
            num_samplers=self.num_samplers,
            actor_critic=cast(ActorCriticModel, self.actor_critic),
        )

        if visualizer is not None:
            assert visualizer.empty()

        num_paused = self.initialize_rollouts(rollouts, visualizer=visualizer)
        num_tasks = sum(
            self.vector_tasks.command(
                "sampler_attr", ["total_unique"] * (self.num_samplers - num_paused)
            )
        )
        # get_logger().debug(
        #     "worker {} number of tasks {}".format(self.worker_id, num_tasks)
        # )
        steps = 0

        self.actor_critic.eval()

        last_time: float = time.time()
        init_time: float = last_time
        frames: int = 0
        if self.mode == "test":
            get_logger().info(
                "worker {}: running evaluation on {} tasks".format(
                    self.worker_id, num_tasks,
                )
            )

        logging_pkg = LoggingPackage(mode=self.mode, training_steps=total_steps)
        while num_paused < self.num_samplers:
            frames += self.num_samplers - num_paused
            num_paused += self.collect_rollout_step(rollouts, visualizer=visualizer)
            steps += 1

            if steps % rollout_steps == 0:
                rollouts.after_update()

            cur_time = time.time()
            if num_paused >= self.num_samplers or cur_time - last_time >= update_secs:
                self.aggregate_task_metrics(logging_pkg=logging_pkg)

                if self.mode == "test":
                    lengths = self.vector_tasks.command(
                        "sampler_attr", ["length"] * (self.num_samplers - num_paused)
                    )
                    npending = sum(lengths)
                    time_to_complete = (
                        "{:.2f}".format(
                            (
                                (cur_time - init_time)
                                * (npending / (num_tasks - npending))
                                / 60
                            )
                        )
                        if npending != num_tasks
                        else "???"
                    )
                    get_logger().info(
                        "worker {}: {:.1f} fps, {}/{} tasks pending ({}). ~{} min. to complete.".format(
                            self.worker_id,
                            frames / (cur_time - init_time),
                            npending,
                            num_tasks,
                            lengths,
                            time_to_complete,
                        )
                    )
                    if logging_pkg.num_non_empty_metrics_dicts_added != 0:
                        get_logger().info(
                            ", ".join(
                                [
                                    "worker {}: num_test_tasks_complete {}".format(
                                        self.worker_id,
                                        logging_pkg.num_non_empty_metrics_dicts_added,
                                    ),
                                    *[
                                        "{} {:.3g}".format(k, v)
                                        for k, v in logging_pkg.metrics_tracker.means().items()
                                    ],
                                ]
                            )
                        )

                    last_time = cur_time

        get_logger().info(
            "worker {}: {} complete, all task samplers paused".format(
                self.mode, self.worker_id
            )
        )

        self.vector_tasks.resume_all()
        self.vector_tasks.set_seeds(self.worker_seeds(self.num_samplers, self.seed))
        self.vector_tasks.reset_all()

        self.aggregate_task_metrics(logging_pkg=logging_pkg)

        logging_pkg.viz_data = (
            visualizer.read_and_reset() if visualizer is not None else None
        )
        logging_pkg.checkpoint_file_name = checkpoint_file_name

        return logging_pkg

    @staticmethod
    def skip_to_latest(checkpoints_queue: mp.Queue, command: Optional[str], data):
        assert (
            checkpoints_queue is not None
        ), "Attempting to process checkpoints queue but this queue is `None`."
        cond = True
        while cond:
            sentinel = ("skip.AUTO.sentinel", time.time())
            checkpoints_queue.put(
                sentinel
            )  # valid since a single valid process is the only consumer
            forwarded = False
            while not forwarded:
                new_command: Optional[str]
                new_data: Any
                (
                    new_command,
                    new_data,
                ) = checkpoints_queue.get()  # block until next command arrives
                if new_command == command:
                    data = new_data
                elif new_command == sentinel[0]:
                    assert (
                        new_data == sentinel[1]
                    ), "wrong sentinel found: {} vs {}".format(new_data, sentinel[1])
                    forwarded = True
                else:
                    raise ValueError(
                        "Unexpected command {} with data {}".format(
                            new_command, new_data
                        )
                    )
            time.sleep(1)
            cond = not checkpoints_queue.empty()
        return data

    def process_checkpoints(self):
        assert (
            self.mode != "train"
        ), "process_checkpoints only to be called from a valid or test instance"

        assert (
            self.checkpoints_queue is not None
        ), "Attempting to process checkpoints queue but this queue is `None`."

        visualizer: Optional[VizSuite] = None

        finalized = False
        try:
            while True:
                command: Optional[str]
                data: Any
                (
                    command,
                    data,
                ) = self.checkpoints_queue.get()  # block until first command arrives
                # get_logger().debug(
                #     "{} {} command {} data {}".format(
                #         self.mode, self.worker_id, command, data
                #     )
                # )

                if command == "eval":
                    if self.num_samplers > 0:
                        if self.mode == "valid":
                            # skip to latest using
                            # 1. there's only consumer in valid
                            # 2. there's no quit/exit/close message issued by runner nor trainer
                            data = self.skip_to_latest(
                                checkpoints_queue=self.checkpoints_queue,
                                command=command,
                                data=data,
                            )

                        if (
                            visualizer is None
                            and self.machine_params.visualizer is not None
                        ):
                            visualizer = self.machine_params.visualizer

                        eval_package = self.run_eval(
                            checkpoint_file_name=data, visualizer=visualizer
                        )

                        self.results_queue.put(eval_package)

                        if self.is_distributed:
                            dist.barrier()
                    else:
                        self.results_queue.put(
                            LoggingPackage(mode=self.mode, training_steps=None,)
                        )
                elif command in ["quit", "exit", "close"]:
                    finalized = True
                    break
                else:
                    raise NotImplementedError()
        except KeyboardInterrupt:
            get_logger().info(
                "KeyboardInterrupt. Terminating {} worker {}".format(
                    self.mode, self.worker_id
                )
            )
        except Exception:
            get_logger().error(
                "Encountered Exception. Terminating {} worker {}".format(
                    self.mode, self.worker_id
                )
            )
            get_logger().error(traceback.format_exc())
        finally:
            if finalized:
                if self.mode == "test":
                    self.results_queue.put(("test_stopped", 0))
                get_logger().info(
                    "{} worker {} complete".format(self.mode, self.worker_id)
                )
            else:
                if self.mode == "test":
                    self.results_queue.put(("test_stopped", self.worker_id + 1))
            self.close(verbose=self.mode == "test")
