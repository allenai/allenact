"""Defines the reinforcement learning `OnPolicyRLEngine`."""

import datetime
import logging
import numbers
import os
import random
import time
import traceback
from functools import partial
from multiprocessing.context import BaseContext
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import filelock
import torch
import torch.distributed as dist  # type: ignore
import torch.distributions  # type: ignore
import torch.multiprocessing as mp  # type: ignore
import torch.nn as nn
import torch.optim as optim

# noinspection PyProtectedMember
from torch._C._distributed_c10d import ReduceOp

from allenact.algorithms.onpolicy_sync.misc import TrackingInfo, TrackingInfoType
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.misc_utils import str2bool
from allenact.utils.model_utils import md5_hash_of_state_dict

try:
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from torch.optim.lr_scheduler import _LRScheduler
except (ImportError, ModuleNotFoundError):
    raise ImportError("`_LRScheduler` was not found in `torch.optim.lr_scheduler`")

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
from allenact.algorithms.onpolicy_sync.storage import (
    ExperienceStorage,
    MiniBatchStorageMixin,
    RolloutStorage,
    StreamingStorageMixin,
)
from allenact.algorithms.onpolicy_sync.vector_sampled_tasks import (
    COMPLETE_TASK_CALLBACK_KEY,
    COMPLETE_TASK_METRICS_KEY,
    SingleProcessVectorSampledTasks,
    VectorSampledTasks,
)
from allenact.base_abstractions.distributions import TeacherForcingDistr
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.misc import (
    ActorCriticOutput,
    GenericAbstractLoss,
    Memory,
    RLStepResult,
)
from allenact.utils import spaces_utils as su
from allenact.utils.experiment_utils import (
    LoggingPackage,
    PipelineStage,
    ScalarMeanTracker,
    StageComponent,
    TrainingPipeline,
    set_deterministic_cudnn,
    set_seed,
    download_checkpoint_from_wandb,
)
from allenact.utils.system import get_logger
from allenact.utils.tensor_utils import batch_observations, detach_recursively
from allenact.utils.viz_utils import VizSuite

try:
    # When debugging we don't want to timeout in the VectorSampledTasks

    # noinspection PyPackageRequirements
    import pydevd

    DEBUGGING = str2bool(os.getenv("ALLENACT_DEBUG", "true"))
except ImportError:
    DEBUGGING = str2bool(os.getenv("ALLENACT_DEBUG", "false"))

DEBUG_VST_TIMEOUT: Optional[int] = (lambda x: int(x) if x is not None else x)(
    os.getenv("ALLENACT_DEBUG_VST_TIMEOUT", None)
)

TRAIN_MODE_STR = "train"
VALID_MODE_STR = "valid"
TEST_MODE_STR = "test"


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
        callback_sensors: Optional[Sequence[Sensor]] = None,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        mp_ctx: Optional[BaseContext] = None,
        worker_id: int = 0,
        num_workers: int = 1,
        device: Union[str, torch.device, int] = "cpu",
        distributed_ip: str = "127.0.0.1",
        distributed_port: int = 0,
        deterministic_agents: bool = False,
        max_sampler_processes_per_worker: Optional[int] = None,
        initial_model_state_dict: Optional[Union[Dict[str, Any], int]] = None,
        try_restart_after_task_error: bool = False,
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
        deterministic_cudnn : Whether to use deterministic cudnn. If `True` this may lower
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

        if self.device != torch.device("cpu"):
            torch.cuda.set_device(device)

        self.distributed_ip = distributed_ip
        self.distributed_port = distributed_port
        self.try_restart_after_task_error = try_restart_after_task_error

        self.mode = mode.lower().strip()
        assert self.mode in [
            TRAIN_MODE_STR,
            VALID_MODE_STR,
            TEST_MODE_STR,
        ], f"Only {TRAIN_MODE_STR}, {VALID_MODE_STR}, {TEST_MODE_STR}, modes supported"

        self.callback_sensors = callback_sensors
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

        self._vector_tasks: Optional[
            Union[VectorSampledTasks, SingleProcessVectorSampledTasks]
        ] = None

        self.sensor_preprocessor_graph = None
        self.actor_critic: Optional[ActorCriticModel] = None

        create_model_kwargs = {}
        if self.machine_params.sensor_preprocessor_graph is not None:
            self.sensor_preprocessor_graph = (
                self.machine_params.sensor_preprocessor_graph.to(self.device)
            )
            create_model_kwargs["sensor_preprocessor_graph"] = (
                self.sensor_preprocessor_graph
            )

        set_seed(self.seed)
        self.actor_critic = cast(
            ActorCriticModel,
            self.config.create_model(**create_model_kwargs),
        ).to(self.device)

        if initial_model_state_dict is not None:
            if isinstance(initial_model_state_dict, int):
                assert (
                    md5_hash_of_state_dict(self.actor_critic.state_dict())
                    == initial_model_state_dict
                ), (
                    f"Could not reproduce the correct model state dict on worker {self.worker_id} despite seeding."
                    f" Please ensure that your model's initialization is reproducable when `set_seed(...)`"
                    f"] has been called with a fixed seed before initialization."
                )
            else:
                self.actor_critic.load_state_dict(
                    state_dict=cast(
                        "OrderedDict[str, Tensor]", initial_model_state_dict
                    )
                )
        else:
            assert mode != TRAIN_MODE_STR or self.num_workers == 1, (
                "When training with multiple workers you must pass a,"
                " non-`None` value for the `initial_model_state_dict` argument."
            )

        if get_logger().level == logging.DEBUG:
            model_hash = md5_hash_of_state_dict(self.actor_critic.state_dict())
            get_logger().debug(
                f"[{self.mode} worker {self.worker_id}] model weights hash: {model_hash}"
            )

        self.is_distributed = False
        self.store: Optional[torch.distributed.TCPStore] = None  # type:ignore
        if self.num_workers > 1:
            self.store = torch.distributed.TCPStore(  # type:ignore
                host_name=self.distributed_ip,
                port=self.distributed_port,
                world_size=self.num_workers,
                is_master=self.worker_id == 0,
                timeout=datetime.timedelta(
                    seconds=3 * (DEBUG_VST_TIMEOUT if DEBUGGING else 1 * 60) + 300
                ),
            )
            cpu_device = self.device == torch.device("cpu")  # type:ignore

            # "gloo" required during testing to ensure that `barrier()` doesn't time out.
            backend = "gloo" if cpu_device or self.mode == TEST_MODE_STR else "nccl"
            get_logger().debug(
                f"Worker {self.worker_id}: initializing distributed {backend} backend with device {self.device}."
            )
            dist.init_process_group(  # type:ignore
                backend=backend,
                store=self.store,
                rank=self.worker_id,
                world_size=self.num_workers,
                # During testing, we sometimes found that default timeout was too short
                # resulting in the run terminating surprisingly, we increase it here.
                timeout=(
                    datetime.timedelta(minutes=3000)
                    if (self.mode == TEST_MODE_STR or DEBUGGING)
                    else dist.default_pg_timeout
                ),
            )
            self.is_distributed = True

        self.deterministic_agents = deterministic_agents

        self._is_closing: bool = (
            False  # Useful for letting the RL runner know if this is closing
        )
        self._is_closed: bool = False

        # Keeping track of metrics and losses during training/inference
        self.single_process_metrics: List = []
        self.single_process_task_callback_data: List = []
        self.tracking_info_list: List[TrackingInfo] = []

        # Variables that wil only be instantiated in the trainer
        self.optimizer: Optional[optim.optimizer.Optimizer] = None
        # noinspection PyProtectedMember
        self.lr_scheduler: Optional[_LRScheduler] = None
        self.insufficient_data_for_update: Optional[torch.distributed.PrefixStore] = (
            None
        )

        # Training pipeline will be instantiated during training and inference.
        # During inference however, it will be instantiated anew on each run of `run_eval`
        # and will be set to `None` after the eval run is complete.
        self.training_pipeline: Optional[TrainingPipeline] = None

    @property
    def vector_tasks(
        self,
    ) -> Union[VectorSampledTasks, SingleProcessVectorSampledTasks]:
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

            # TODO: The `self.max_sampler_processes_per_worker == 1` case below would be
            #   great to have but it does not play nicely with us wanting to kill things
            #   using SIGTERM/SIGINT signals. Would be nice to figure out a solution to
            #   this at some point.
            # if self.max_sampler_processes_per_worker == 1:
            #     # No need to instantiate a new task sampler processes if we're
            #     # restricted to one sampler process for this worker.
            #     self._vector_tasks = SingleProcessVectorSampledTasks(
            #         make_sampler_fn=self.config.make_sampler_fn,
            #         sampler_fn_args_list=self.get_sampler_fn_args(seeds),
            #     )
            # else:
            self._vector_tasks = VectorSampledTasks(
                make_sampler_fn=self.config.make_sampler_fn,
                sampler_fn_args=self.get_sampler_fn_args(seeds),
                callback_sensors=self.callback_sensors,
                multiprocessing_start_method=(
                    "forkserver" if self.mp_ctx is None else None
                ),
                mp_ctx=self.mp_ctx,
                max_processes=self.max_sampler_processes_per_worker,
                read_timeout=DEBUG_VST_TIMEOUT if DEBUGGING else 1 * 60,
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
        seeds = [random.randint(0, (2**31) - 1) for _ in range(nprocesses)]
        if initial_seed is not None:
            random.setstate(rstate)
        return seeds

    def get_sampler_fn_args(self, seeds: Optional[List[int]] = None):
        sampler_devices = self.machine_params.sampler_devices

        if self.mode == TRAIN_MODE_STR:
            fn = self.config.train_task_sampler_args
        elif self.mode == VALID_MODE_STR:
            fn = self.config.valid_task_sampler_args
        elif self.mode == TEST_MODE_STR:
            fn = self.config.test_task_sampler_args
        else:
            raise NotImplementedError(
                f"self.mode must be one of {TRAIN_MODE_STR}, {VALID_MODE_STR}, or {TEST_MODE_STR}."
            )

        if self.is_distributed:
            total_processes = sum(self.num_samplers_per_worker)
            process_offset = sum(self.num_samplers_per_worker[: self.worker_id])
        else:
            total_processes = self.num_samplers
            process_offset = 0

        sampler_devices_as_ints: Optional[List[int]] = None
        if (
            self.is_distributed or self.mode == TEST_MODE_STR
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
        self, ckpt: Union[str, Dict[str, Any]], restart_pipeline: bool
    ) -> Dict[str, Union[Dict[str, Any], torch.Tensor, float, int, str, List]]:
        if isinstance(ckpt, str):
            get_logger().info(
                f"[{self.mode} worker {self.worker_id}] Loading checkpoint from {ckpt}"
            )
            # Map location CPU is almost always better than mapping to a CUDA device.
            ckpt = torch.load(os.path.abspath(ckpt), map_location="cpu")

        ckpt = cast(
            Dict[str, Union[Dict[str, Any], torch.Tensor, float, int, str, List]],
            ckpt,
        )

        self.actor_critic.load_state_dict(ckpt["model_state_dict"])  # type:ignore

        if "training_pipeline_state_dict" in ckpt and not restart_pipeline:
            self.training_pipeline.load_state_dict(
                cast(Dict[str, Any], ckpt["training_pipeline_state_dict"])
            )

        return ckpt

    # aggregates task metrics currently in queue
    def aggregate_task_metrics(
        self,
        logging_pkg: LoggingPackage,
        num_tasks: int = -1,
    ) -> LoggingPackage:
        if num_tasks > 0:
            if len(self.single_process_metrics) != num_tasks:
                error_msg = (
                    "shorter"
                    if len(self.single_process_metrics) < num_tasks
                    else "longer"
                )
                get_logger().error(
                    f"Metrics out is {error_msg} than expected number of tasks."
                    " This should only happen if a positive number of `num_tasks` were"
                    " set during testing but the queue did not contain this number of entries."
                    " Please file an issue at https://github.com/allenai/allenact/issues."
                )

        num_empty_tasks_dequeued = 0

        for metrics_dict in self.single_process_metrics:
            num_empty_tasks_dequeued += not logging_pkg.add_metrics_dict(
                single_task_metrics_dict=metrics_dict
            )

        self.single_process_metrics = []

        if num_empty_tasks_dequeued != 0:
            get_logger().warning(
                f"Discarded {num_empty_tasks_dequeued} empty task metrics"
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

    def initialize_storage_and_viz(
        self,
        storage_to_initialize: Optional[Sequence[ExperienceStorage]],
        visualizer: Optional[VizSuite] = None,
    ):
        keep: Optional[List] = None
        if visualizer is not None or (
            storage_to_initialize is not None
            and any(isinstance(s, RolloutStorage) for s in storage_to_initialize)
        ):
            # No rollout storage, thus we are not
            observations = self.vector_tasks.get_observations()

            npaused, keep, batch = self.remove_paused(observations)
            observations = (
                self._preprocess_observations(batch) if len(keep) > 0 else batch
            )

            assert npaused == 0, f"{npaused} samplers are paused during initialization."

            num_samplers = len(keep)
        else:
            observations = {}
            num_samplers = 0
            npaused = 0

        recurrent_memory_specification = (
            self.actor_critic.recurrent_memory_specification
        )

        if storage_to_initialize is not None:
            for s in storage_to_initialize:
                s.to(self.device)
                s.set_partition(index=self.worker_id, num_parts=self.num_workers)
                s.initialize(
                    observations=observations,
                    num_samplers=num_samplers,
                    recurrent_memory_specification=recurrent_memory_specification,
                    action_space=self.actor_critic.action_space,
                )

        if visualizer is not None and num_samplers > 0:
            visualizer.collect(vector_task=self.vector_tasks, alive=keep)

        return npaused

    @property
    def num_active_samplers(self):
        if self.vector_tasks is None:
            return 0
        return self.vector_tasks.num_unpaused_tasks

    def act(
        self,
        rollout_storage: RolloutStorage,
        dist_wrapper_class: Optional[type] = None,
    ):
        with torch.no_grad():
            agent_input = rollout_storage.agent_input_for_next_step()
            actor_critic_output, memory = self.actor_critic(**agent_input)

            distr = actor_critic_output.distributions
            if dist_wrapper_class is not None:
                distr = dist_wrapper_class(distr=distr, obs=agent_input["observations"])

            actions = distr.sample() if not self.deterministic_agents else distr.mode()

        return actions, actor_critic_output, memory, agent_input["observations"]

    def aggregate_and_send_logging_package(
        self,
        tracking_info_list: List[TrackingInfo],
        logging_pkg: Optional[LoggingPackage] = None,
        send_logging_package: bool = True,
        checkpoint_file_name: Optional[str] = None,
    ):
        if logging_pkg is None:
            logging_pkg = LoggingPackage(
                mode=self.mode,
                training_steps=self.training_pipeline.total_steps,
                pipeline_stage=self.training_pipeline.current_stage_index,
                storage_uuid_to_total_experiences=self.training_pipeline.storage_uuid_to_total_experiences,
                checkpoint_file_name=checkpoint_file_name,
            )

        self.aggregate_task_metrics(logging_pkg=logging_pkg)

        for callback_dict in self.single_process_task_callback_data:
            logging_pkg.task_callback_data.append(callback_dict)
        self.single_process_task_callback_data = []

        for tracking_info in tracking_info_list:
            if tracking_info.n < 0:
                get_logger().warning(
                    f"Obtained a train_info_dict with {tracking_info.n} elements."
                    f" Full info: ({tracking_info.type}, {tracking_info.info}, {tracking_info.n})."
                )
            else:
                tracking_info_dict = tracking_info.info

                if tracking_info.type == TrackingInfoType.LOSS:
                    tracking_info_dict = {
                        f"losses/{k}": v for k, v in tracking_info_dict.items()
                    }

                logging_pkg.add_info_dict(
                    info_dict=tracking_info_dict,
                    n=tracking_info.n,
                    stage_component_uuid=tracking_info.stage_component_uuid,
                    storage_uuid=tracking_info.storage_uuid,
                )

        if send_logging_package:
            self.results_queue.put(logging_pkg)

        return logging_pkg

    @staticmethod
    def _active_memory(memory, keep):
        return memory.sampler_select(keep) if memory is not None else memory

    def probe(self, dones: List[bool], npaused, period=100000):
        """Debugging util. When called from
        self.collect_step_across_all_task_samplers(...), calls render for the
        0-th task sampler of the 0-th distributed worker for the first
        beginning episode spaced at least period steps from the beginning of
        the previous one.

        For valid, train, it currently renders all episodes for the 0-th task sampler of the
        0-th distributed worker. If this is not wanted, it must be hard-coded for now below.

        # Parameters

        dones : dones list from self.collect_step_across_all_task_samplers(...)
        npaused : number of newly paused tasks returned by self.removed_paused(...)
        period : minimal spacing in sampled steps between the beginning of episodes to be shown.
        """
        sampler_id = 0
        done = dones[sampler_id]
        if self.mode != TRAIN_MODE_STR:
            setattr(
                self, "_probe_npaused", getattr(self, "_probe_npaused", 0) + npaused
            )
            if self._probe_npaused == self.num_samplers:  # type:ignore
                del self._probe_npaused  # type:ignore
                return
            period = 0
        if self.worker_id == 0:
            if done:
                if period > 0 and (
                    getattr(self, "_probe_steps", None) is None
                    or (
                        self._probe_steps < 0  # type:ignore
                        and (
                            self.training_pipeline.total_steps
                            + self._probe_steps  # type:ignore
                        )
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
                    # noinspection PyAttributeOutsideInit
                    self._probe_steps = -self._probe_steps

    def collect_step_across_all_task_samplers(
        self,
        rollout_storage_uuid: str,
        uuid_to_storage: Dict[str, ExperienceStorage],
        visualizer=None,
        dist_wrapper_class=None,
    ) -> int:
        rollout_storage = cast(RolloutStorage, uuid_to_storage[rollout_storage_uuid])
        actions, actor_critic_output, memory, _ = self.act(
            rollout_storage=rollout_storage,
            dist_wrapper_class=dist_wrapper_class,
        )

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
            if step_result.info is not None:
                if COMPLETE_TASK_METRICS_KEY in step_result.info:
                    self.single_process_metrics.append(
                        step_result.info[COMPLETE_TASK_METRICS_KEY]
                    )
                    del step_result.info[COMPLETE_TASK_METRICS_KEY]
                if COMPLETE_TASK_CALLBACK_KEY in step_result.info:
                    self.single_process_task_callback_data.append(
                        step_result.info[COMPLETE_TASK_CALLBACK_KEY]
                    )
                    del step_result.info[COMPLETE_TASK_CALLBACK_KEY]

        rewards: Union[List, torch.Tensor]
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        rewards = torch.tensor(
            rewards,
            dtype=torch.float,
            device=self.device,  # type:ignore
        )

        # We want rewards to have dimensions [sampler, reward]
        if len(rewards.shape) == 1:
            # Rewards are of shape [sampler,]
            rewards = rewards.unsqueeze(-1)
        elif len(rewards.shape) > 1:
            raise NotImplementedError()

        # If done then clean the history of observations.
        masks = (
            1.0
            - torch.tensor(
                dones,
                dtype=torch.float32,
                device=self.device,  # type:ignore
            )
        ).view(
            -1, 1
        )  # [sampler, 1]

        npaused, keep, batch = self.remove_paused(observations)

        if hasattr(self.actor_critic, "sampler_select"):
            self.actor_critic.sampler_select(keep)

        # TODO self.probe(...) can be useful for debugging (we might want to control it from main?)
        # self.probe(dones, npaused)

        if npaused > 0:
            if self.mode == TRAIN_MODE_STR:
                raise NotImplementedError(
                    "When trying to get a new task from a task sampler (using the `.next_task()` method)"
                    " the task sampler returned `None`. This is not currently supported during training"
                    " (and almost certainly a bug in the implementation of the task sampler or in the "
                    " initialization of the task sampler for training)."
                )

            for s in uuid_to_storage.values():
                if isinstance(s, RolloutStorage):
                    s.sampler_select(keep)

        to_add_to_storage = dict(
            observations=(
                self._preprocess_observations(batch) if len(keep) > 0 else batch
            ),
            memory=self._active_memory(memory, keep),
            actions=flat_actions[0, keep],
            action_log_probs=actor_critic_output.distributions.log_prob(actions)[
                0, keep
            ],
            value_preds=actor_critic_output.values[0, keep],
            rewards=rewards[keep],
            masks=masks[keep],
        )
        for storage in uuid_to_storage.values():
            storage.add(**to_add_to_storage)

        # TODO we always miss tensors for the last action in the last episode of each worker
        if visualizer is not None:
            if len(keep) > 0:
                visualizer.collect(
                    rollout=rollout_storage,
                    vector_task=self.vector_tasks,
                    alive=keep,
                    actor_critic=actor_critic_output,
                )
            else:
                visualizer.collect(actor_critic=actor_critic_output)

        return npaused

    def distributed_weighted_sum(
        self,
        to_share: Union[torch.Tensor, float, int],
        weight: Union[torch.Tensor, float, int],
    ):
        """Weighted sum of scalar across distributed workers."""
        if self.is_distributed:
            aggregate = torch.tensor(to_share * weight).to(self.device)
            dist.all_reduce(aggregate)
            return aggregate.item()
        else:
            if abs(1 - weight) > 1e-5:
                get_logger().warning(
                    f"Scaling non-distributed value with weight {weight}"
                )
            return torch.tensor(to_share * weight).item()

    def distributed_reduce(
        self, to_share: Union[torch.Tensor, float, int], op: ReduceOp
    ):
        """Weighted sum of scalar across distributed workers."""
        if self.is_distributed:
            aggregate = torch.tensor(to_share).to(self.device)
            dist.all_reduce(aggregate, op=op)
            return aggregate.item()
        else:
            return torch.tensor(to_share).item()

    def backprop_step(
        self,
        total_loss: torch.Tensor,
        max_grad_norm: float,
        local_to_global_batch_size_ratio: float = 1.0,
    ):
        raise NotImplementedError

    def save_error_data(self, batch: Dict[str, Any]):
        raise NotImplementedError

    @property
    def step_count(self) -> int:
        if (
            self.training_pipeline.current_stage is None
        ):  # Might occur during testing when all stages are complete
            return 0
        return self.training_pipeline.current_stage.steps_taken_in_stage

    def compute_losses_track_them_and_backprop(
        self,
        stage: PipelineStage,
        stage_component: StageComponent,
        storage: ExperienceStorage,
        skip_backprop: bool = False,
    ):
        training = self.mode == TRAIN_MODE_STR

        assert training or skip_backprop

        if training and self.is_distributed:
            self.insufficient_data_for_update.set(
                "insufficient_data_for_update", str(0)
            )
            dist.barrier(
                device_ids=(
                    None if self.device == torch.device("cpu") else [self.device.index]
                )
            )

        training_settings = stage_component.training_settings

        loss_names = stage_component.loss_names
        losses = [self.training_pipeline.get_loss(ln) for ln in loss_names]
        loss_weights = [stage.uuid_to_loss_weight[ln] for ln in loss_names]
        loss_update_repeats_list = training_settings.update_repeats
        if isinstance(loss_update_repeats_list, numbers.Integral):
            loss_update_repeats_list = [loss_update_repeats_list] * len(loss_names)

        if skip_backprop and isinstance(storage, MiniBatchStorageMixin):
            if loss_update_repeats_list != [1] * len(loss_names):
                loss_update_repeats_list = [1] * len(loss_names)
                get_logger().warning(
                    "Does not make sense to do multiple updates when"
                    " skip_backprop is `True` and you are using a storage of type"
                    " `MiniBatchStorageMixin`. This is likely a problem caused by"
                    " using a custom valid/test stage component that is inheriting its"
                    " TrainingSettings from the TrainingPipeline's TrainingSettings. We will override"
                    " the requested number of updates repeats (which was"
                    f" {dict(zip(loss_names, loss_update_repeats_list))}) to be 1 for all losses."
                )

        enough_data_for_update = True
        for current_update_repeat_index in range(
            max(loss_update_repeats_list, default=0)
        ):
            if isinstance(storage, MiniBatchStorageMixin):
                batch_iterator = storage.batched_experience_generator(
                    num_mini_batch=training_settings.num_mini_batch
                )
            elif isinstance(storage, StreamingStorageMixin):
                assert (
                    training_settings.num_mini_batch is None
                    or training_settings.num_mini_batch == 1
                )

                def single_batch_generator(streaming_storage: StreamingStorageMixin):
                    try:
                        yield cast(
                            StreamingStorageMixin, streaming_storage
                        ).next_batch()
                    except EOFError:
                        if not training:
                            raise

                        if streaming_storage.empty():
                            yield None
                        else:
                            cast(
                                StreamingStorageMixin, streaming_storage
                            ).reset_stream()
                            stage.stage_component_uuid_to_stream_memory[
                                stage_component.uuid
                            ].clear()
                            yield cast(
                                StreamingStorageMixin, streaming_storage
                            ).next_batch()

                batch_iterator = single_batch_generator(streaming_storage=storage)
            else:
                raise NotImplementedError(
                    f"Storage {storage} must be a subclass of `MiniBatchStorageMixin` or `StreamingStorageMixin`."
                )

            for batch in batch_iterator:
                if batch is None:
                    # This should only happen in a `StreamingStorageMixin` when it cannot
                    # generate an initial batch or when we are in testing/validation and
                    # we've reached the end of the dataset over which to test/validate.
                    if training:
                        assert isinstance(storage, StreamingStorageMixin)
                        get_logger().warning(
                            f"Worker {self.worker_id}: could not run update in {storage}, potentially because"
                            f" not enough data has been accumulated to be able to fill an initial batch."
                        )
                    else:
                        pass
                    enough_data_for_update = False

                if training and self.is_distributed:
                    self.insufficient_data_for_update.add(
                        "insufficient_data_for_update",
                        1 * (not enough_data_for_update),
                    )
                    dist.barrier(
                        device_ids=(
                            None
                            if self.device == torch.device("cpu")
                            else [self.device.index]
                        )
                    )

                    if (
                        int(
                            self.insufficient_data_for_update.get(
                                "insufficient_data_for_update"
                            )
                        )
                        != 0
                    ):
                        enough_data_for_update = False
                        break

                info: Dict[str, float] = {}

                bsize: Optional[int] = None
                total_loss: Optional[torch.Tensor] = None
                actor_critic_output_for_batch: Optional[ActorCriticOutput] = None
                batch_memory = Memory()

                for loss, loss_name, loss_weight, max_update_repeats_for_loss in zip(
                    losses, loss_names, loss_weights, loss_update_repeats_list
                ):
                    if current_update_repeat_index >= max_update_repeats_for_loss:
                        continue

                    if isinstance(loss, AbstractActorCriticLoss):
                        bsize = batch["bsize"]

                        if actor_critic_output_for_batch is None:
                            try:
                                actor_critic_output_for_batch, _ = self.actor_critic(
                                    observations=batch["observations"],
                                    memory=batch["memory"],
                                    prev_actions=batch["prev_actions"],
                                    masks=batch["masks"],
                                )
                            except ValueError:
                                save_path = self.save_error_data(batch=batch)
                                get_logger().error(
                                    f"Encountered a value error! Likely because of nans in the output/input."
                                    f" Saving all error information to {save_path}."
                                )
                                raise

                        loss_return = loss.loss(
                            step_count=self.step_count,
                            batch=batch,
                            actor_critic_output=actor_critic_output_for_batch,
                        )

                        per_epoch_info = {}
                        if len(loss_return) == 2:
                            current_loss, current_info = loss_return
                        elif len(loss_return) == 3:
                            current_loss, current_info, per_epoch_info = loss_return
                        else:
                            raise NotImplementedError

                    elif isinstance(loss, GenericAbstractLoss):
                        loss_output = loss.loss(
                            model=self.actor_critic,
                            batch=batch,
                            batch_memory=batch_memory,
                            stream_memory=stage.stage_component_uuid_to_stream_memory[
                                stage_component.uuid
                            ],
                        )
                        current_loss = loss_output.value
                        current_info = loss_output.info
                        per_epoch_info = loss_output.per_epoch_info
                        batch_memory = loss_output.batch_memory
                        stage.stage_component_uuid_to_stream_memory[
                            stage_component.uuid
                        ] = loss_output.stream_memory
                        bsize = loss_output.bsize
                    else:
                        raise NotImplementedError(
                            f"Loss of type {type(loss)} is not supported. Losses must be subclasses of"
                            f" `AbstractActorCriticLoss` or `GenericAbstractLoss`."
                        )

                    if total_loss is None:
                        total_loss = loss_weight * current_loss
                    else:
                        total_loss = total_loss + loss_weight * current_loss

                    for key, value in current_info.items():
                        info[f"{loss_name}/{key}"] = value

                    if per_epoch_info is not None:
                        for key, value in per_epoch_info.items():
                            if max(loss_update_repeats_list, default=0) > 1:
                                info[
                                    f"{loss_name}/{key}_epoch{current_update_repeat_index:02d}"
                                ] = value
                                info[f"{loss_name}/{key}_combined"] = value
                            else:
                                info[f"{loss_name}/{key}"] = value

                assert total_loss is not None, (
                    f"No {stage_component.uuid} losses specified for training in stage"
                    f" {self.training_pipeline.current_stage_index}"
                )

                total_loss_scalar = total_loss.item()
                info[f"total_loss"] = total_loss_scalar

                self.tracking_info_list.append(
                    TrackingInfo(
                        type=TrackingInfoType.LOSS,
                        info=info,
                        n=bsize,
                        storage_uuid=stage_component.storage_uuid,
                        stage_component_uuid=stage_component.uuid,
                    )
                )

                to_track = {
                    "rollout_epochs": max(loss_update_repeats_list, default=0),
                    "worker_batch_size": bsize,
                }

                aggregate_bsize = None
                if training:
                    aggregate_bsize = self.distributed_weighted_sum(bsize, 1)
                    to_track["global_batch_size"] = aggregate_bsize
                    to_track["lr"] = self.optimizer.param_groups[0]["lr"]

                if training_settings.num_mini_batch is not None:
                    to_track["rollout_num_mini_batch"] = (
                        training_settings.num_mini_batch
                    )

                for k, v in to_track.items():
                    # We need to set the bsize to 1 for `worker_batch_size` below as we're trying to record the
                    # average batch size per worker, not the average per worker weighted by the size of the batches
                    # of those workers.
                    self.tracking_info_list.append(
                        TrackingInfo(
                            type=TrackingInfoType.UPDATE_INFO,
                            info={k: v},
                            n=1 if k == "worker_batch_size" else bsize,
                            storage_uuid=stage_component.storage_uuid,
                            stage_component_uuid=stage_component.uuid,
                        )
                    )

                if not skip_backprop:
                    total_grad_norm = self.backprop_step(
                        total_loss=total_loss,
                        max_grad_norm=training_settings.max_grad_norm,
                        local_to_global_batch_size_ratio=bsize / aggregate_bsize,
                    )
                    self.tracking_info_list.append(
                        TrackingInfo(
                            type=TrackingInfoType.UPDATE_INFO,
                            info={"total_grad_norm": total_grad_norm},
                            n=bsize,
                            storage_uuid=stage_component.storage_uuid,
                            stage_component_uuid=stage_component.uuid,
                        )
                    )

                stage.stage_component_uuid_to_stream_memory[stage_component.uuid] = (
                    detach_recursively(
                        input=stage.stage_component_uuid_to_stream_memory[
                            stage_component.uuid
                        ],
                        inplace=True,
                    )
                )

    def close(self, verbose=True):
        self._is_closing = True

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
                    f"[{self.mode} worker {self.worker_id}] Closing OnPolicyRLEngine.vector_tasks."
                )
                self._vector_tasks.close()
                logif(f"[{self.mode} worker {self.worker_id}] Closed.")
            except Exception as e:
                logif(
                    f"[{self.mode} worker {self.worker_id}] Exception raised when closing OnPolicyRLEngine.vector_tasks:"
                )
                logif(e)

        self._is_closed = True
        self._is_closing = False

    @property
    def is_closed(self):
        return self._is_closed

    @property
    def is_closing(self):
        return self._is_closing

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
        distributed_ip: str = "127.0.0.1",
        distributed_port: int = 0,
        deterministic_agents: bool = False,
        distributed_preemption_threshold: float = 0.7,
        max_sampler_processes_per_worker: Optional[int] = None,
        save_ckpt_after_every_pipeline_stage: bool = True,
        first_local_worker_id: int = 0,
        **kwargs,
    ):
        kwargs["mode"] = TRAIN_MODE_STR
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
            distributed_ip=distributed_ip,
            distributed_port=distributed_port,
            deterministic_agents=deterministic_agents,
            max_sampler_processes_per_worker=max_sampler_processes_per_worker,
            **kwargs,
        )

        self.save_ckpt_after_every_pipeline_stage = save_ckpt_after_every_pipeline_stage

        self.actor_critic.train()

        self.training_pipeline: TrainingPipeline = config.training_pipeline()

        if self.num_workers != 1:
            # Ensure that we're only using early stopping criterions in the non-distributed setting.
            if any(
                stage.early_stopping_criterion is not None
                for stage in self.training_pipeline.pipeline_stages
            ):
                raise NotImplementedError(
                    "Early stopping criterions are currently only allowed when using a single training worker, i.e."
                    " no distributed (multi-GPU) training. If this is a feature you'd like please create an issue"
                    " at https://github.com/allenai/allenact/issues or (even better) create a pull request with this "
                    " feature and we'll be happy to review it."
                )

        self.optimizer: optim.optimizer.Optimizer = (
            self.training_pipeline.optimizer_builder(
                params=[p for p in self.actor_critic.parameters() if p.requires_grad]
            )
        )

        # noinspection PyProtectedMember
        self.lr_scheduler: Optional[_LRScheduler] = None
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
            # Flag for finished worker in current epoch with custom component
            self.insufficient_data_for_update = (
                torch.distributed.PrefixStore(  # type:ignore
                    "insufficient_data_for_update", self.store
                )
            )
        else:
            self.num_workers_done = None
            self.num_workers_steps = None
            self.distributed_preemption_threshold = 1.0
            self.offpolicy_epoch_done = None

        # Keeping track of training state
        self.former_steps: Optional[int] = None
        self.last_log: Optional[int] = None
        self.last_save: Optional[int] = None
        # The `self._last_aggregated_train_task_metrics` attribute defined
        # below is used for early stopping criterion computations
        self._last_aggregated_train_task_metrics: ScalarMeanTracker = (
            ScalarMeanTracker()
        )

        self.first_local_worker_id = first_local_worker_id

    def advance_seed(
        self, seed: Optional[int], return_same_seed_per_worker=False
    ) -> Optional[int]:
        if seed is None:
            return seed
        seed = (seed ^ (self.training_pipeline.total_steps + 1)) % (
            2**31 - 1
        )  # same seed for all workers

        if (not return_same_seed_per_worker) and (
            self.mode == TRAIN_MODE_STR or self.mode == TEST_MODE_STR
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
            )  # use the latest seed for workers and update rng state
            if self.vector_tasks is not None:
                self.vector_tasks.set_seeds(seeds)

    def save_error_data(self, batch: Dict[str, Any]) -> str:
        model_path = os.path.join(
            self.checkpoints_dir,
            "error_for_exp_{}__stage_{:02d}__steps_{:012d}.pt".format(
                self.experiment_name,
                self.training_pipeline.current_stage_index,
                self.training_pipeline.total_steps,
            ),
        )
        with filelock.FileLock(
            os.path.join(self.checkpoints_dir, "error.lock"), timeout=60
        ):
            if not os.path.exists(model_path):
                save_dict = {
                    "model_state_dict": self.actor_critic.state_dict(),  # type:ignore
                    "total_steps": self.training_pipeline.total_steps,  # Total steps including current stage
                    "optimizer_state_dict": self.optimizer.state_dict(),  # type: ignore
                    "training_pipeline_state_dict": self.training_pipeline.state_dict(),
                    "trainer_seed": self.seed,
                    "batch": batch,
                }

                if self.lr_scheduler is not None:
                    save_dict["scheduler_state"] = cast(
                        _LRScheduler, self.lr_scheduler
                    ).state_dict()

                torch.save(save_dict, model_path)
        return model_path

    def aggregate_and_send_logging_package(
        self,
        tracking_info_list: List[TrackingInfo],
        logging_pkg: Optional[LoggingPackage] = None,
        send_logging_package: bool = True,
        checkpoint_file_name: Optional[str] = None,
    ):
        logging_pkg = super().aggregate_and_send_logging_package(
            tracking_info_list=tracking_info_list,
            logging_pkg=logging_pkg,
            send_logging_package=send_logging_package,
            checkpoint_file_name=checkpoint_file_name,
        )

        if self.mode == TRAIN_MODE_STR:
            # Technically self.mode should always be "train" here (as this is the training engine),
            # this conditional is defensive
            self._last_aggregated_train_task_metrics.add_scalars(
                scalars=logging_pkg.metrics_tracker.means(),
                n=logging_pkg.metrics_tracker.counts(),
            )

        return logging_pkg

    def checkpoint_save(self, pipeline_stage_index: Optional[int] = None) -> str:
        model_path = os.path.join(
            self.checkpoints_dir,
            "exp_{}__stage_{:02d}__steps_{:012d}.pt".format(
                self.experiment_name,
                (
                    self.training_pipeline.current_stage_index
                    if pipeline_stage_index is None
                    else pipeline_stage_index
                ),
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
        if restart_pipeline:
            if "training_pipeline_state_dict" in ckpt:
                del ckpt["training_pipeline_state_dict"]

        ckpt = super().checkpoint_load(ckpt, restart_pipeline=restart_pipeline)

        if restart_pipeline:
            self.training_pipeline.restart_pipeline()
        else:
            self.seed = cast(int, ckpt["trainer_seed"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # type: ignore
            if self.lr_scheduler is not None and "scheduler_state" in ckpt:
                self.lr_scheduler.load_state_dict(ckpt["scheduler_state"])  # type: ignore

        self.deterministic_seeds()

        return ckpt

    @property
    def step_count(self):
        return self.training_pipeline.current_stage.steps_taken_in_stage

    @step_count.setter
    def step_count(self, val: int) -> None:
        self.training_pipeline.current_stage.steps_taken_in_stage = val

    @property
    def log_interval(self):
        return (
            self.training_pipeline.current_stage.training_settings.metric_accumulate_interval
        )

    @property
    def approx_steps(self):
        if self.is_distributed:
            # the actual number of steps gets synchronized after each rollout
            return (
                self.step_count - self.former_steps
            ) * self.num_workers + self.former_steps
        else:
            return self.step_count  # this is actually accurate

    def act(
        self,
        rollout_storage: RolloutStorage,
        dist_wrapper_class: Optional[type] = None,
    ):
        if self.training_pipeline.current_stage.teacher_forcing is not None:
            assert dist_wrapper_class is None

            def tracking_callback(type: TrackingInfoType, info: Dict[str, Any], n: int):
                self.tracking_info_list.append(
                    TrackingInfo(
                        type=type,
                        info=info,
                        n=n,
                        storage_uuid=self.training_pipeline.rollout_storage_uuid,
                        stage_component_uuid=None,
                    )
                )

            dist_wrapper_class = partial(
                TeacherForcingDistr,
                action_space=self.actor_critic.action_space,
                num_active_samplers=self.num_active_samplers,
                approx_steps=self.approx_steps,
                teacher_forcing=self.training_pipeline.current_stage.teacher_forcing,
                tracking_callback=tracking_callback,
            )

        actions, actor_critic_output, memory, step_observation = super().act(
            rollout_storage=rollout_storage,
            dist_wrapper_class=dist_wrapper_class,
        )

        self.step_count += self.num_active_samplers

        return actions, actor_critic_output, memory, step_observation

    def advantage_stats(self, advantages: torch.Tensor) -> Dict[str, torch.Tensor]:
        r"""Computes the mean and variances of advantages (possibly over multiple workers).
        For multiple workers, this method is equivalent to first collecting all versions of
        advantages and then computing the mean and variance locally over that.

        # Parameters

        advantages: Tensors to compute mean and variance over. Assumed to be solely the
         worker's local copy of this tensor, the resultant mean and variance will be computed
         as though _all_ workers' versions of this tensor were concatenated together in
         distributed training.
        """

        # Step count has already been updated with the steps from all workers
        global_rollout_steps = self.step_count - self.former_steps

        if self.is_distributed:
            summed_advantages = advantages.sum()
            dist.all_reduce(summed_advantages)
            mean = summed_advantages / global_rollout_steps

            summed_squares = (advantages - mean).pow(2).sum()
            dist.all_reduce(summed_squares)
            std = (summed_squares / (global_rollout_steps - 1)).sqrt()
        else:
            # noinspection PyArgumentList
            mean, std = advantages.mean(), advantages.std()

        return {"mean": mean, "std": std}

    def backprop_step(
        self,
        total_loss: torch.Tensor,
        max_grad_norm: float,
        local_to_global_batch_size_ratio: float = 1.0,
    ):
        self.optimizer.zero_grad()  # type: ignore
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()

        if self.is_distributed:
            # From https://github.com/pytorch/pytorch/issues/43135
            reductions, all_params = [], []
            for p in self.actor_critic.parameters():
                # you can also organize grads to larger buckets to make all_reduce more efficient
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    else:  # local_global_batch_size_tuple is not None, since we're distributed:
                        p.grad = p.grad * local_to_global_batch_size_ratio
                    reductions.append(
                        dist.all_reduce(
                            p.grad,
                            async_op=True,
                        )  # sum
                    )  # synchronize
                    all_params.append(p)
            for reduction, p in zip(reductions, all_params):
                reduction.wait()

        if hasattr(self.actor_critic, "compute_total_grad_norm"):
            total_grad_norm = self.actor_critic.compute_total_grad_norm().item()
        else:
            total_grad_norm = 0.0

        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(),
            max_norm=max_grad_norm,  # type: ignore
        )

        self.optimizer.step()  # type: ignore
        return total_grad_norm

    def _save_checkpoint_then_send_checkpoint_for_validation_and_update_last_save_counter(
        self, pipeline_stage_index: Optional[int] = None
    ):
        model_path = None
        self.deterministic_seeds()
        # if self.worker_id == self.first_local_worker_id:
        if self.worker_id == 0:
            model_path = self.checkpoint_save(pipeline_stage_index=pipeline_stage_index)
            if self.checkpoints_queue is not None:
                self.checkpoints_queue.put(("eval", model_path))
        self.last_save = self.training_pipeline.total_steps
        return model_path

    def run_pipeline(self, valid_on_initial_weights: bool = False):
        cur_stage_training_settings = (
            self.training_pipeline.current_stage.training_settings
        )

        # Change engine attributes that depend on the current stage
        self.training_pipeline.current_stage.change_engine_attributes(self)

        rollout_storage = self.training_pipeline.rollout_storage
        uuid_to_storage = self.training_pipeline.current_stage_storage
        self.initialize_storage_and_viz(
            storage_to_initialize=cast(
                List[ExperienceStorage], list(uuid_to_storage.values())
            )
        )
        self.tracking_info_list.clear()

        self.last_log = self.training_pipeline.total_steps

        if self.last_save is None:
            self.last_save = self.training_pipeline.total_steps

        should_save_checkpoints = (
            self.checkpoints_dir != ""
            and cur_stage_training_settings.save_interval is not None
            and cur_stage_training_settings.save_interval > 0
        )
        already_saved_checkpoint = False

        if (
            valid_on_initial_weights
            and should_save_checkpoints
            and self.checkpoints_queue is not None
        ):
            # if self.worker_id == self.first_local_worker_id:
            if self.worker_id == 0:
                model_path = self.checkpoint_save()
                if self.checkpoints_queue is not None:
                    self.checkpoints_queue.put(("eval", model_path))

        while True:
            pipeline_stage_changed = self.training_pipeline.before_rollout(
                train_metrics=self._last_aggregated_train_task_metrics
            )  # This is `False` at the very start of training, i.e. pipeline starts with a stage initialized

            self._last_aggregated_train_task_metrics.reset()
            training_is_complete = self.training_pipeline.current_stage is None

            # `training_is_complete` should imply `pipeline_stage_changed`
            assert pipeline_stage_changed or not training_is_complete

            #  Saving checkpoints and initializing storage when the pipeline stage changes
            if pipeline_stage_changed:
                # Here we handle saving a checkpoint after a pipeline stage ends. We
                # do this:
                # (1) after every pipeline stage if the `self.save_ckpt_after_every_pipeline_stage`
                #   boolean is True, and
                # (2) when we have reached the end of ALL training (i.e. all stages are complete).
                if (
                    should_save_checkpoints
                    and (  # Might happen if the `save_interval` was hit just previously, see below
                        not already_saved_checkpoint
                    )
                    and (
                        self.save_ckpt_after_every_pipeline_stage
                        or training_is_complete
                    )
                ):
                    self._save_checkpoint_then_send_checkpoint_for_validation_and_update_last_save_counter(
                        pipeline_stage_index=(
                            self.training_pipeline.current_stage_index - 1
                            if not training_is_complete
                            else len(self.training_pipeline.pipeline_stages) - 1
                        )
                    )

                # If training is complete, break out
                if training_is_complete:
                    break

                # Here we handle updating our training settings after a pipeline stage ends.
                # Update the training settings we're using
                cur_stage_training_settings = (
                    self.training_pipeline.current_stage.training_settings
                )

                # If the pipeline stage changed we must initialize any new custom storage and
                # stop updating any custom storage that is no longer in use (this second bit
                # is done by simply updating `uuid_to_storage` to the new custom storage objects).
                new_uuid_to_storage = self.training_pipeline.current_stage_storage
                storage_to_initialize = [
                    s
                    for uuid, s in new_uuid_to_storage.items()
                    if uuid
                    not in uuid_to_storage  # Don't initialize storage already in use
                ]
                self.initialize_storage_and_viz(
                    storage_to_initialize=storage_to_initialize,
                )
                uuid_to_storage = new_uuid_to_storage

                # Change engine attributes that depend on the current stage
                self.training_pipeline.current_stage.change_engine_attributes(self)

            already_saved_checkpoint = False

            if self.is_distributed:
                self.num_workers_done.set("done", str(0))
                self.num_workers_steps.set("steps", str(0))
                # Ensure all workers are done before incrementing num_workers_{steps, done}
                dist.barrier(
                    device_ids=(
                        None
                        if self.device == torch.device("cpu")
                        else [self.device.index]
                    )
                )

            self.former_steps = self.step_count
            former_storage_experiences = {
                k: v.total_experiences
                for k, v in self.training_pipeline.current_stage_storage.items()
            }

            if self.training_pipeline.rollout_storage_uuid is None:
                # In this case we're not expecting to collect storage experiences, i.e. everything
                # will be off-policy.

                # self.step_count is normally updated by the `self.collect_step_across_all_task_samplers`
                # call below, but since we're not collecting onpolicy experiences, we need to update
                # it here. The step count here is now just effectively a count of the number of times
                # we've called `compute_losses_track_them_and_backprop` below.
                self.step_count += 1

                before_update_info = dict(
                    next_value=None,
                    use_gae=cur_stage_training_settings.use_gae,
                    gamma=cur_stage_training_settings.gamma,
                    tau=cur_stage_training_settings.gae_lambda,
                    adv_stats_callback=self.advantage_stats,
                )
            else:
                vector_tasks_already_restarted = False
                step = -1
                while step < cur_stage_training_settings.num_steps - 1:
                    step += 1

                    try:
                        num_paused = self.collect_step_across_all_task_samplers(
                            rollout_storage_uuid=self.training_pipeline.rollout_storage_uuid,
                            uuid_to_storage=uuid_to_storage,
                        )
                    except (TimeoutError, EOFError) as e:
                        if (
                            not self.try_restart_after_task_error
                        ) or self.mode != TRAIN_MODE_STR:
                            # Apparently you can just call `raise` here and doing so will just raise the exception as though
                            # it was not caught (so the stacktrace isn't messed up)
                            raise
                        elif vector_tasks_already_restarted:
                            raise RuntimeError(
                                f"[{self.mode} worker {self.worker_id}] `vector_tasks` has timed out twice in the same"
                                f" rollout. This suggests that this error was not recoverable. Timeout exception:\n{traceback.format_exc()}"
                            )
                        else:
                            get_logger().warning(
                                f"[{self.mode} worker {self.worker_id}] `vector_tasks` appears to have crashed during"
                                f" training due to an {type(e).__name__} error. You have set"
                                f" `try_restart_after_task_error` to `True` so we will attempt to restart these tasks from"
                                f" the beginning. USE THIS FEATURE AT YOUR OWN"
                                f" RISK. Exception:\n{traceback.format_exc()}."
                            )
                            self.vector_tasks.close()
                            self._vector_tasks = None

                            vector_tasks_already_restarted = True
                            for (
                                storage
                            ) in self.training_pipeline.current_stage_storage.values():
                                storage.after_updates()
                            self.initialize_storage_and_viz(
                                storage_to_initialize=cast(
                                    List[ExperienceStorage],
                                    list(uuid_to_storage.values()),
                                )
                            )
                            step = -1
                            continue

                    # A more informative error message should already have been thrown in be given in
                    # `collect_step_across_all_task_samplers` if `num_paused != 0` here but this serves
                    # as a sanity check.
                    assert num_paused == 0

                    if self.is_distributed:
                        # Preempt stragglers
                        # Each worker will stop collecting steps for the current rollout whenever a
                        # 100 * distributed_preemption_threshold percentage of workers are finished collecting their
                        # rollout steps, and we have collected at least 25% but less than 90% of the steps.
                        num_done = int(self.num_workers_done.get("done"))
                        if (
                            num_done
                            > self.distributed_preemption_threshold * self.num_workers
                            and 0.25 * cur_stage_training_settings.num_steps
                            <= step
                            < 0.9 * cur_stage_training_settings.num_steps
                        ):
                            get_logger().debug(
                                f"[{self.mode} worker {self.worker_id}] Preempted after {step}"
                                f" steps (out of {cur_stage_training_settings.num_steps})"
                                f" with {num_done} workers done"
                            )
                            break

                with torch.no_grad():
                    actor_critic_output, _ = self.actor_critic(
                        **rollout_storage.agent_input_for_next_step()
                    )

                self.training_pipeline.rollout_count += 1

                if self.is_distributed:
                    # Mark that a worker is done collecting experience
                    self.num_workers_done.add("done", 1)
                    self.num_workers_steps.add(
                        "steps", self.step_count - self.former_steps
                    )

                    # Ensure all workers are done before updating step counter
                    dist.barrier(
                        device_ids=(
                            None
                            if self.device == torch.device("cpu")
                            else [self.device.index]
                        )
                    )

                    ndone = int(self.num_workers_done.get("done"))
                    assert (
                        ndone == self.num_workers
                    ), f"# workers done {ndone} != # workers {self.num_workers}"

                    # get the actual step_count
                    self.step_count = (
                        int(self.num_workers_steps.get("steps")) + self.former_steps
                    )

                before_update_info = dict(
                    next_value=actor_critic_output.values.detach(),
                    use_gae=cur_stage_training_settings.use_gae,
                    gamma=cur_stage_training_settings.gamma,
                    tau=cur_stage_training_settings.gae_lambda,
                    adv_stats_callback=self.advantage_stats,
                )

            # Prepare storage for iteration during updates
            for storage in self.training_pipeline.current_stage_storage.values():
                storage.before_updates(**before_update_info)

            for sc in self.training_pipeline.current_stage.stage_components:
                component_storage = uuid_to_storage[sc.storage_uuid]

                self.compute_losses_track_them_and_backprop(
                    stage=self.training_pipeline.current_stage,
                    stage_component=sc,
                    storage=component_storage,
                )

            for storage in self.training_pipeline.current_stage_storage.values():
                storage.after_updates()

            # We update the storage step counts saved in
            # `self.training_pipeline.current_stage.storage_uuid_to_steps_taken_in_stage` here rather than with
            # `self.steps` above because some storage step counts may only change after the update calls above.
            # This may seem a bit weird but consider a storage that corresponds to a fixed dataset
            # used for imitation learning. For such a dataset, the "steps" will only increase as
            # new batches are sampled during update calls.
            # Note: We don't need to sort the keys below to ensure that distributed updates happen correctly
            #   as `self.training_pipeline.current_stage_storage` is an ordered `dict`.
            # First we calculate the change in counts (possibly aggregating across devices)
            change_in_storage_experiences = {}
            for k in sorted(self.training_pipeline.current_stage_storage.keys()):
                delta = (
                    self.training_pipeline.current_stage_storage[k].total_experiences
                    - former_storage_experiences[k]
                )
                assert delta >= 0
                change_in_storage_experiences[k] = self.distributed_weighted_sum(
                    to_share=delta, weight=1
                )

            # Then we update `self.training_pipeline.current_stage.storage_uuid_to_steps_taken_in_stage` with the above
            # computed changes.
            for storage_uuid, delta in change_in_storage_experiences.items():
                self.training_pipeline.current_stage.storage_uuid_to_steps_taken_in_stage[
                    storage_uuid
                ] += delta

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch=self.training_pipeline.total_steps)

            # Here we handle saving a checkpoint every `save_interval` steps, saving after
            # a pipeline stage completes is controlled above
            checkpoint_file_name = None
            if should_save_checkpoints and (
                    self.training_pipeline.total_steps - self.last_save
                    >= cur_stage_training_settings.save_interval
            ):
                checkpoint_file_name = self._save_checkpoint_then_send_checkpoint_for_validation_and_update_last_save_counter()
                already_saved_checkpoint = True

            if (
                self.training_pipeline.total_steps - self.last_log >= self.log_interval
                or self.training_pipeline.current_stage.is_complete
            ):
                self.aggregate_and_send_logging_package(
                    tracking_info_list=self.tracking_info_list,
                    checkpoint_file_name=checkpoint_file_name,
                )
                self.tracking_info_list.clear()
                self.last_log = self.training_pipeline.total_steps

            if (
                cur_stage_training_settings.advance_scene_rollout_period is not None
            ) and (
                self.training_pipeline.rollout_count
                % cur_stage_training_settings.advance_scene_rollout_period
                == 0
            ):
                get_logger().info(
                    f"[{self.mode} worker {self.worker_id}] Force advance"
                    f" tasks with {self.training_pipeline.rollout_count} rollouts"
                )
                self.vector_tasks.next_task(force_advance_scene=True)
                self.initialize_storage_and_viz(
                    storage_to_initialize=cast(
                        List[ExperienceStorage], list(uuid_to_storage.values())
                    )
                )

    def train(
        self,
        checkpoint_file_name: Optional[str] = None,
        restart_pipeline: bool = False,
        valid_on_initial_weights: bool = False,
    ):
        assert (
            self.mode == TRAIN_MODE_STR
        ), "train only to be called from a train instance"

        training_completed_successfully = False
        # noinspection PyBroadException
        try:
            if checkpoint_file_name is not None:
                if "wandb://" == checkpoint_file_name[:8]:
                    ckpt_dir = "wandb_ckpts"
                    os.makedirs(ckpt_dir, exist_ok=True)
                    checkpoint_file_name = download_checkpoint_from_wandb(
                        checkpoint_path_dir_or_pattern,
                        ckpt_dir,
                        only_allow_one_ckpt=True
                    )
                self.checkpoint_load(checkpoint_file_name, restart_pipeline)

            self.run_pipeline(valid_on_initial_weights=valid_on_initial_weights)

            training_completed_successfully = True
        except KeyboardInterrupt:
            get_logger().info(
                f"[{self.mode} worker {self.worker_id}] KeyboardInterrupt, exiting."
            )
        except Exception as e:
            get_logger().error(
                f"[{self.mode} worker {self.worker_id}] Encountered {type(e).__name__}, exiting."
            )
            get_logger().error(traceback.format_exc())
        finally:
            if training_completed_successfully:
                if self.worker_id == 0:
                    self.results_queue.put(("train_stopped", 0))
                get_logger().info(
                    f"[{self.mode} worker {self.worker_id}] Training finished successfully."
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
        enforce_expert: bool = False,
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

        self.enforce_expert = enforce_expert

    def run_eval(
        self,
        checkpoint_file_path: str,
        rollout_steps: int = 100,
        visualizer: Optional[VizSuite] = None,
        update_secs: float = 20.0,
        verbose: bool = False,
    ) -> LoggingPackage:
        assert self.actor_critic is not None, "called `run_eval` with no actor_critic"

        # Sanity check that we haven't entered an invalid state. During eval the training_pipeline
        # should be only set in this function and always unset at the end of it.
        assert self.training_pipeline is None, (
            "`training_pipeline` should be `None` before calling `run_eval`."
            " This is necessary as we want to initialize new storages."
        )
        self.training_pipeline = self.config.training_pipeline()

        ckpt = self.checkpoint_load(checkpoint_file_path, restart_pipeline=False)
        total_steps = cast(int, ckpt["total_steps"])

        eval_pipeline_stage = cast(
            PipelineStage,
            getattr(self.training_pipeline, f"{self.mode}_pipeline_stage"),
        )
        assert (
            len(eval_pipeline_stage.stage_components) <= 1
        ), "Only one StageComponent is supported during inference."
        uuid_to_storage = self.training_pipeline.get_stage_storage(eval_pipeline_stage)

        assert len(uuid_to_storage) > 0, (
            "No storage found for eval pipeline stage, this is a bug in AllenAct,"
            " please submit an issue on GitHub (https://github.com/allenai/allenact/issues)."
        )

        uuid_to_rollout_storage = {
            uuid: storage
            for uuid, storage in uuid_to_storage.items()
            if isinstance(storage, RolloutStorage)
        }
        uuid_to_non_rollout_storage = {
            uuid: storage
            for uuid, storage in uuid_to_storage.items()
            if not isinstance(storage, RolloutStorage)
        }

        if len(uuid_to_rollout_storage) > 1 or len(uuid_to_non_rollout_storage) > 1:
            raise NotImplementedError(
                "Only one RolloutStorage and non-RolloutStorage object is allowed within an evaluation pipeline stage."
                " If you'd like to evaluate against multiple storages please"
                " submit an issue on GitHub (https://github.com/allenai/allenact/issues). For the moment you'll need"
                " to evaluate against these storages separately."
            )

        rollout_storage = self.training_pipeline.rollout_storage

        if visualizer is not None:
            assert visualizer.empty()

        num_paused = self.initialize_storage_and_viz(
            storage_to_initialize=cast(
                List[ExperienceStorage], list(uuid_to_storage.values())
            ),
            visualizer=visualizer,
        )
        assert num_paused == 0, f"{num_paused} tasks paused when initializing eval"

        if rollout_storage is not None:
            num_tasks = sum(
                self.vector_tasks.command(
                    "sampler_attr", ["length"] * self.num_active_samplers
                )
            ) + (  # We need to add this as the first tasks have already been sampled
                self.num_active_samplers
            )
        else:
            num_tasks = 0

        # get_logger().debug("worker {self.worker_id} number of tasks {num_tasks}")
        steps = 0

        self.actor_critic.eval()

        last_time: float = time.time()
        init_time: float = last_time
        frames: int = 0
        if verbose:
            get_logger().info(
                f"[{self.mode} worker {self.worker_id}] Running evaluation on {num_tasks} tasks"
                f" for ckpt {checkpoint_file_path}"
            )

        if self.enforce_expert:
            dist_wrapper_class = partial(
                TeacherForcingDistr,
                action_space=self.actor_critic.action_space,
                num_active_samplers=None,
                approx_steps=None,
                teacher_forcing=None,
                tracking_callback=None,
                always_enforce=True,
            )
        else:
            dist_wrapper_class = None

        logging_pkg = LoggingPackage(
            mode=self.mode,
            training_steps=total_steps,
            storage_uuid_to_total_experiences=self.training_pipeline.storage_uuid_to_total_experiences,
        )
        should_compute_onpolicy_losses = (
            len(eval_pipeline_stage.loss_names) > 0
            and eval_pipeline_stage.stage_components[0].storage_uuid
            == self.training_pipeline.rollout_storage_uuid
        )
        while self.num_active_samplers > 0:
            frames += self.num_active_samplers
            num_newly_paused = self.collect_step_across_all_task_samplers(
                rollout_storage_uuid=self.training_pipeline.rollout_storage_uuid,
                uuid_to_storage=uuid_to_rollout_storage,
                visualizer=visualizer,
                dist_wrapper_class=dist_wrapper_class,
            )
            steps += 1

            if should_compute_onpolicy_losses and num_newly_paused > 0:
                # The `collect_step_across_all_task_samplers` method will automatically drop
                # parts of the rollout storage that correspond to paused tasks (namely by calling"
                # `rollout_storage.sampler_select(UNPAUSED_TASK_INDS)`). This makes sense when you don't need to
                # compute losses for tasks but is a bit limiting here as we're throwing away data before
                # using it to compute losses. As changing this is non-trivial we'll just warn the user
                # for now.
                get_logger().warning(
                    f"[{self.mode} worker {self.worker_id}] {num_newly_paused * rollout_storage.step} steps"
                    f" will be dropped when computing losses in evaluation. This is a limitation of the current"
                    f" implementation of rollout collection in AllenAct. If you'd like to see this"
                    f" functionality improved please submit an issue on GitHub"
                    f" (https://github.com/allenai/allenact/issues)."
                )

            if self.num_active_samplers == 0 or steps % rollout_steps == 0:
                if should_compute_onpolicy_losses and self.num_active_samplers > 0:
                    with torch.no_grad():
                        actor_critic_output, _ = self.actor_critic(
                            **rollout_storage.agent_input_for_next_step()
                        )
                        before_update_info = dict(
                            next_value=actor_critic_output.values.detach(),
                            use_gae=eval_pipeline_stage.training_settings.use_gae,
                            gamma=eval_pipeline_stage.training_settings.gamma,
                            tau=eval_pipeline_stage.training_settings.gae_lambda,
                            adv_stats_callback=lambda advantages: {
                                "mean": advantages.mean(),
                                "std": advantages.std(),
                            },
                        )
                    # Prepare storage for iteration during loss computation
                    for storage in uuid_to_rollout_storage.values():
                        storage.before_updates(**before_update_info)

                    # Compute losses
                    with torch.no_grad():
                        for sc in eval_pipeline_stage.stage_components:
                            self.compute_losses_track_them_and_backprop(
                                stage=eval_pipeline_stage,
                                stage_component=sc,
                                storage=uuid_to_rollout_storage[sc.storage_uuid],
                                skip_backprop=True,
                            )

                for storage in uuid_to_rollout_storage.values():
                    storage.after_updates()

            cur_time = time.time()
            if self.num_active_samplers == 0 or cur_time - last_time >= update_secs:
                logging_pkg = self.aggregate_and_send_logging_package(
                    tracking_info_list=self.tracking_info_list,
                    logging_pkg=logging_pkg,
                    send_logging_package=False,
                )
                self.tracking_info_list.clear()

                if verbose:
                    npending: int
                    lengths: List[int]
                    if self.num_active_samplers > 0:
                        lengths = self.vector_tasks.command(
                            "sampler_attr",
                            ["length"] * self.num_active_samplers,
                        )
                        npending = sum(lengths)
                    else:
                        lengths = []
                        npending = 0
                    est_time_to_complete = (
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
                        f"[{self.mode} worker {self.worker_id}]"
                        f" For ckpt {checkpoint_file_path}"
                        f" {frames / (cur_time - init_time):.1f} fps,"
                        f" {npending}/{num_tasks} tasks pending ({lengths})."
                        f" ~{est_time_to_complete} min. to complete."
                    )
                    if logging_pkg.num_non_empty_metrics_dicts_added != 0:
                        get_logger().info(
                            ", ".join(
                                [
                                    f"[{self.mode} worker {self.worker_id}]"
                                    f" num_{self.mode}_tasks_complete {logging_pkg.num_non_empty_metrics_dicts_added}",
                                    *[
                                        f"{k} {v:.3g}"
                                        for k, v in logging_pkg.metrics_tracker.means().items()
                                    ],
                                    *[
                                        f"{k0[1]}/{k1} {v1:.3g}"
                                        for k0, v0 in logging_pkg.info_trackers.items()
                                        for k1, v1 in v0.means().items()
                                    ],
                                ]
                            )
                        )

                    last_time = cur_time

        get_logger().info(
            f"[{self.mode} worker {self.worker_id}] Task evaluation complete, all task samplers paused."
        )

        if rollout_storage is not None:
            self.vector_tasks.resume_all()
            self.vector_tasks.set_seeds(self.worker_seeds(self.num_samplers, self.seed))
            self.vector_tasks.reset_all()

        logging_pkg = self.aggregate_and_send_logging_package(
            tracking_info_list=self.tracking_info_list,
            logging_pkg=logging_pkg,
            send_logging_package=False,
        )
        self.tracking_info_list.clear()

        logging_pkg.viz_data = (
            visualizer.read_and_reset() if visualizer is not None else None
        )

        should_compute_offpolicy_losses = (
            len(eval_pipeline_stage.loss_names) > 0
            and not should_compute_onpolicy_losses
        )
        if should_compute_offpolicy_losses:
            # In this case we are evaluating a non-rollout storage, e.g. some off-policy data
            get_logger().info(
                f"[{self.mode} worker {self.worker_id}] Non-rollout storage detected, will now compute losses"
                f" using this storage."
            )

            offpolicy_eval_done = False
            while not offpolicy_eval_done:
                before_update_info = dict(
                    next_value=None,
                    use_gae=eval_pipeline_stage.training_settings.use_gae,
                    gamma=eval_pipeline_stage.training_settings.gamma,
                    tau=eval_pipeline_stage.training_settings.gae_lambda,
                    adv_stats_callback=lambda advantages: {
                        "mean": advantages.mean(),
                        "std": advantages.std(),
                    },
                )
                # Prepare storage for iteration during loss computation
                for storage in uuid_to_non_rollout_storage.values():
                    storage.before_updates(**before_update_info)

                # Compute losses
                assert len(eval_pipeline_stage.stage_components) == 1
                try:
                    for sc in eval_pipeline_stage.stage_components:
                        with torch.no_grad():
                            self.compute_losses_track_them_and_backprop(
                                stage=eval_pipeline_stage,
                                stage_component=sc,
                                storage=uuid_to_non_rollout_storage[sc.storage_uuid],
                                skip_backprop=True,
                            )
                except EOFError:
                    offpolicy_eval_done = True

                for storage in uuid_to_non_rollout_storage.values():
                    storage.after_updates()

                total_bsize = sum(
                    tif.info.get("worker_batch_size", 0)
                    for tif in self.tracking_info_list
                )
                logging_pkg = self.aggregate_and_send_logging_package(
                    tracking_info_list=self.tracking_info_list,
                    logging_pkg=logging_pkg,
                    send_logging_package=False,
                )
                self.tracking_info_list.clear()

                cur_time = time.time()
                if verbose and (cur_time - last_time >= update_secs):
                    get_logger().info(
                        f"[{self.mode} worker {self.worker_id}]"
                        f" For ckpt {checkpoint_file_path}"
                        f" {total_bsize / (cur_time - init_time):.1f} its/sec."
                    )
                    if logging_pkg.info_trackers != 0:
                        get_logger().info(
                            ", ".join(
                                [
                                    f"[{self.mode} worker {self.worker_id}]"
                                    f" num_{self.mode}_iters_complete {total_bsize}",
                                    *[
                                        f"{'/'.join(k0)}/{k1} {v1:.3g}"
                                        for k0, v0 in logging_pkg.info_trackers.items()
                                        for k1, v1 in v0.means().items()
                                    ],
                                ]
                            )
                        )

                    last_time = cur_time

        # Call after_updates here to reset all storages
        for storage in uuid_to_storage.values():
            storage.after_updates()

        # Set the training pipeline to `None` so that the storages do not
        # persist across calls to `run_eval`
        self.training_pipeline = None

        logging_pkg.checkpoint_file_name = checkpoint_file_path

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
                    ), f"Wrong sentinel found: {new_data} vs {sentinel[1]}"
                    forwarded = True
                else:
                    raise ValueError(
                        f"Unexpected command {new_command} with data {new_data}"
                    )
            time.sleep(1)
            cond = not checkpoints_queue.empty()
        return data

    def process_checkpoints(self):
        assert (
            self.mode != TRAIN_MODE_STR
        ), "process_checkpoints only to be called from a valid or test instance"

        assert (
            self.checkpoints_queue is not None
        ), "Attempting to process checkpoints queue but this queue is `None`."

        visualizer: Optional[VizSuite] = None

        finalized = False
        # noinspection PyBroadException
        try:
            while True:
                command: Optional[str]
                ckp_file_path: Any
                (
                    command,
                    ckp_file_path,
                ) = self.checkpoints_queue.get()  # block until first command arrives
                # get_logger().debug(
                #     "{} {} command {} data {}".format(
                #         self.mode, self.worker_id, command, data
                #     )
                # )

                if command == "eval":
                    if self.mode == VALID_MODE_STR:
                        # skip to latest using
                        # 1. there's only consumer in valid
                        # 2. there's no quit/exit/close message issued by runner nor trainer
                        ckp_file_path = self.skip_to_latest(
                            checkpoints_queue=self.checkpoints_queue,
                            command=command,
                            data=ckp_file_path,
                        )

                    if (
                        visualizer is None
                        and self.machine_params.visualizer is not None
                    ):
                        visualizer = self.machine_params.visualizer

                    eval_package = self.run_eval(
                        checkpoint_file_path=ckp_file_path,
                        visualizer=visualizer,
                        verbose=True,
                        update_secs=20 if self.mode == TEST_MODE_STR else 5 * 60,
                    )

                    self.results_queue.put(eval_package)

                    if self.is_distributed:
                        dist.barrier()
                elif command in ["quit", "exit", "close"]:
                    finalized = True
                    break
                else:
                    raise NotImplementedError()
        except KeyboardInterrupt:
            get_logger().info(
                f"[{self.mode} worker {self.worker_id}] KeyboardInterrupt, exiting."
            )
        except Exception as e:
            get_logger().error(
                f"[{self.mode} worker {self.worker_id}] Encountered {type(e).__name__}, exiting."
            )
            get_logger().error(traceback.format_exc())
        finally:
            if finalized:
                if self.mode == TEST_MODE_STR:
                    self.results_queue.put(("test_stopped", 0))
                get_logger().info(
                    f"[{self.mode} worker {self.worker_id}] Complete, all checkpoints processed."
                )
            else:
                if self.mode == TEST_MODE_STR:
                    self.results_queue.put(("test_stopped", self.worker_id + 1))
            self.close(verbose=self.mode == TEST_MODE_STR)
