"""Defines the reinforcement learning `OnPolicyRunner`."""
import copy
import glob
import itertools
import json
import math
import os
import pathlib
import queue
import random
import signal
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Optional, Dict, Union, Tuple, Sequence, List, Any

import filelock
import numpy as np
import torch
import torch.multiprocessing as mp
from setproctitle import setproctitle as ptitle

from allenact.algorithms.onpolicy_sync.engine import (
    OnPolicyTrainer,
    OnPolicyInference,
    TRAIN_MODE_STR,
    VALID_MODE_STR,
    TEST_MODE_STR,
    OnPolicyRLEngine,
)
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.utils.experiment_utils import (
    ScalarMeanTracker,
    set_deterministic_cudnn,
    set_seed,
    LoggingPackage,
)
from allenact.utils.misc_utils import (
    all_equal,
    get_git_diff_of_project,
    NumpyJSONEncoder,
)
from allenact.utils.model_utils import md5_hash_of_state_dict
from allenact.utils.system import get_logger, find_free_port
from allenact.utils.tensor_utils import SummaryWriter
from allenact.utils.viz_utils import VizSuite

_CONFIG_KWARGS_STR = "__CONFIG_KWARGS__"


# Has results queue (aggregated per trainer), checkpoints queue and mp context
# Instantiates train, validate, and test workers
# Logging
# Saves configs, makes folder for trainer models
class OnPolicyRunner(object):
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str,
        loaded_config_src_files: Optional[Dict[str, str]],
        seed: Optional[int] = None,
        mode: str = "train",
        deterministic_cudnn: bool = False,
        deterministic_agents: bool = False,
        mp_ctx: Optional[BaseContext] = None,
        multiprocessing_start_method: str = "default",
        extra_tag: str = "",
        disable_tensorboard: bool = False,
        disable_config_saving: bool = False,
        distributed_ip_port: str = "127.0.0.1:-1",
        machine_id: int = 0,
    ):
        self.config = config
        self.output_dir = output_dir
        self.loaded_config_src_files = loaded_config_src_files
        self.seed = seed if seed is not None else random.randint(0, 2 ** 31 - 1)
        self.deterministic_cudnn = deterministic_cudnn
        if multiprocessing_start_method == "default":
            if torch.cuda.is_available():
                multiprocessing_start_method = "forkserver"
            else:
                # Spawn seems to play nicer with cpus and debugging
                multiprocessing_start_method = "spawn"
        self.mp_ctx = self.init_context(mp_ctx, multiprocessing_start_method)
        self.extra_tag = extra_tag
        self.mode = mode.lower().strip()
        self.visualizer: Optional[VizSuite] = None
        self.deterministic_agents = deterministic_agents
        self.disable_tensorboard = disable_tensorboard
        self.disable_config_saving = disable_config_saving

        assert self.mode in [
            TRAIN_MODE_STR,
            TEST_MODE_STR,
        ], "Only 'train' and 'test' modes supported in runner"

        if self.deterministic_cudnn:
            set_deterministic_cudnn()

        set_seed(self.seed)

        self.queues: Optional[Dict[str, mp.Queue]] = None

        self.processes: Dict[str, List[Union[BaseProcess, mp.Process]]] = defaultdict(
            list
        )

        self.current_checkpoint = None

        self._local_start_time_str: Optional[str] = None

        self._is_closed: bool = False

        self._collect_valid_results: bool = False

        self.distributed_ip_port = distributed_ip_port
        self.machine_id = machine_id

    @property
    def local_start_time_str(self) -> str:
        if self._local_start_time_str is None:
            raise RuntimeError(
                "Local start time string does not exist as neither `start_train()` or `start_test()`"
                " has been called on this runner."
            )
        return self._local_start_time_str

    @property
    def running_validation(self):
        return (
            sum(
                MachineParams.instance_from(
                    self.config.machine_params(VALID_MODE_STR)
                ).nprocesses
            )
            > 0
        ) and self.machine_id == 0

    @staticmethod
    def init_context(
        mp_ctx: Optional[BaseContext] = None,
        multiprocessing_start_method: str = "forkserver",
        valid_start_methods: Tuple[str, ...] = ("forkserver", "spawn", "fork"),
    ):
        if mp_ctx is None:
            assert multiprocessing_start_method in valid_start_methods, (
                "multiprocessing_start_method must be one of {}. Got '{}'"
            ).format(valid_start_methods, multiprocessing_start_method)

            mp_ctx = mp.get_context(multiprocessing_start_method)
        elif multiprocessing_start_method != mp_ctx.get_start_method():
            get_logger().warning(
                "ignoring multiprocessing_start_method '{}' and using given context with '{}'".format(
                    multiprocessing_start_method, mp_ctx.get_start_method()
                )
            )

        return mp_ctx

    def _acquire_unique_local_start_time_string(self) -> str:
        """Creates a (unique) local start time string for this experiment.

        Ensures through file locks that the local start time string
        produced is unique. This implies that, if one has many
        experiments starting in in parallel, at most one will be started
        every second (as the local start time string only records the
        time up to the current second).
        """
        os.makedirs(self.output_dir, exist_ok=True)
        start_time_string_lock_path = os.path.abspath(
            os.path.join(self.output_dir, ".allenact_start_time_string.lock")
        )
        try:
            with filelock.FileLock(start_time_string_lock_path, timeout=60):
                last_start_time_string_path = os.path.join(
                    self.output_dir, ".allenact_last_start_time_string"
                )
                pathlib.Path(last_start_time_string_path).touch()

                with open(last_start_time_string_path, "r") as f:
                    last_start_time_string_list = f.readlines()

                while True:
                    candidate_str = time.strftime(
                        "%Y-%m-%d_%H-%M-%S", time.localtime(time.time())
                    )
                    if (
                        len(last_start_time_string_list) == 0
                        or last_start_time_string_list[0].strip() != candidate_str
                    ):
                        break
                    time.sleep(0.2)

                with open(last_start_time_string_path, "w") as f:
                    f.write(candidate_str)

        except filelock.Timeout as e:
            get_logger().exception(
                f"Could not acquire the lock for {start_time_string_lock_path} for 60 seconds,"
                " this suggests an unexpected deadlock. Please close all AllenAct training processes,"
                " delete this lockfile, and try again."
            )
            raise e

        assert candidate_str is not None
        return candidate_str

    def worker_devices(self, mode: str):
        machine_params: MachineParams = MachineParams.instance_from(
            self.config.machine_params(mode)
        )
        devices = machine_params.devices

        assert all_equal(devices) or all(
            d.index >= 0 for d in devices
        ), f"Cannot have a mix of CPU and GPU devices (`devices == {devices}`)"

        get_logger().info(
            "Using {} {} workers on devices {}".format(len(devices), mode, devices)
        )
        return devices

    def worker_ids(self, mode: str):
        machine_params: MachineParams = MachineParams.instance_from(
            self.config.machine_params(mode, machine_id=self.machine_id)
        )
        ids = machine_params.ids

        get_logger().info(f"Using worker ids {ids} ({len(ids)} workers in {self.machine_id})")
        return ids

    def init_visualizer(self, mode: str):
        if not self.disable_tensorboard:
            # Note: Avoid instantiating anything in machine_params (use Builder if needed)
            machine_params = MachineParams.instance_from(
                self.config.machine_params(mode)
            )
            self.visualizer = machine_params.visualizer

    @staticmethod
    def init_process(mode: str, id: int, to_close_on_termination: OnPolicyRLEngine):
        ptitle(f"{mode}-{id}")

        def create_handler(termination_type: str):
            def handler(_signo, _frame):
                prefix = f"{termination_type} signal sent to worker {mode}-{id}."
                if to_close_on_termination._is_closed:
                    get_logger().info(
                        f"{prefix} Worker {mode}-{id} is already closed, exiting."
                    )
                    sys.exit(0)
                elif not to_close_on_termination._is_closing:
                    get_logger().info(
                        f"{prefix} Forcing worker {mode}-{id} to close and exiting."
                    )
                    try:
                        to_close_on_termination.close(True)
                    except Exception:
                        get_logger().error(
                            f"Error occurred when closing the RL engine used by work {mode}-{id}."
                            f" We cannot recover from this and will simply exit. The exception:"
                        )
                        get_logger().exception(traceback.format_exc())
                        sys.exit(1)
                    sys.exit(0)
                else:
                    get_logger().info(
                        f"{prefix} Worker {mode}-{id} is already closing, ignoring this signal."
                    )

            return handler

        signal.signal(signal.SIGTERM, create_handler("Termination"))
        signal.signal(signal.SIGINT, create_handler("Interrupt"))

    @staticmethod
    def init_worker(engine_class, args, kwargs):
        mode = kwargs["mode"]
        id = kwargs["worker_id"]

        worker = None
        try:
            worker = engine_class(*args, **kwargs)
        except Exception:
            get_logger().error(
                "Encountered Exception. Terminating {} worker {}".format(mode, id)
            )
            get_logger().exception(traceback.format_exc())
            kwargs["results_queue"].put(("{}_stopped".format(mode), 1 + id))
        finally:
            return worker

    @staticmethod
    def train_loop(
        id: int = 0,
        checkpoint: Optional[str] = None,
        restart_pipeline: bool = False,
        *engine_args,
        **engine_kwargs,
    ):
        engine_kwargs["mode"] = TRAIN_MODE_STR
        engine_kwargs["worker_id"] = id
        engine_kwargs_for_print = {
            k: (v if k != "initial_model_state_dict" else "[SUPPRESSED]")
            for k, v in engine_kwargs.items()
        }
        get_logger().info(f"train {id} args {engine_kwargs_for_print}")

        trainer: OnPolicyTrainer = OnPolicyRunner.init_worker(
            engine_class=OnPolicyTrainer, args=engine_args, kwargs=engine_kwargs
        )
        if trainer is not None:
            OnPolicyRunner.init_process("Train", id, to_close_on_termination=trainer)
            trainer.train(
                checkpoint_file_name=checkpoint, restart_pipeline=restart_pipeline
            )

    @staticmethod
    def valid_loop(id: int = 0, *engine_args, **engine_kwargs):
        engine_kwargs["mode"] = VALID_MODE_STR
        engine_kwargs["worker_id"] = id
        get_logger().info("valid {} args {}".format(id, engine_kwargs))

        valid = OnPolicyRunner.init_worker(
            engine_class=OnPolicyInference, args=engine_args, kwargs=engine_kwargs
        )
        if valid is not None:
            OnPolicyRunner.init_process("Valid", id, to_close_on_termination=valid)
            valid.process_checkpoints()  # gets checkpoints via queue

    @staticmethod
    def test_loop(id: int = 0, *engine_args, **engine_kwargs):
        engine_kwargs["mode"] = TEST_MODE_STR
        engine_kwargs["worker_id"] = id
        get_logger().info("test {} args {}".format(id, engine_kwargs))

        test = OnPolicyRunner.init_worker(OnPolicyInference, engine_args, engine_kwargs)
        if test is not None:
            OnPolicyRunner.init_process("Test", id, to_close_on_termination=test)
            test.process_checkpoints()  # gets checkpoints via queue

    def _initialize_start_train_or_start_test(self):
        self._is_closed = False

        if self.queues is not None:
            for k, q in self.queues.items():
                try:
                    out = q.get(timeout=1)
                    raise RuntimeError(
                        f"{k} queue was not empty before starting new training/testing (contained {out})."
                        f" This should not happen, please report how you obtained this error"
                        f" by creating an issue at https://github.com/allenai/allenact/issues."
                    )
                except queue.Empty:
                    pass

        self.queues = {
            "results": self.mp_ctx.Queue(),
            "checkpoints": self.mp_ctx.Queue(),
        }

        self._local_start_time_str = self._acquire_unique_local_start_time_string()

    def get_port(self):
        passed_port = int(self.distributed_ip_port.split(":")[1])
        if passed_port < 0:
            assert self.machine_id == 0, "Only runner with `machine_id` == 0 can search for a free port."
            distributed_port = find_free_port(self.distributed_ip_port.split(":")[0])
        else:
            distributed_port = passed_port

        get_logger().info(
            f"Engines on machine_id == {self.machine_id} using port {distributed_port} and seed {self.seed}"
        )

        return distributed_port

    def start_train(
        self,
        checkpoint: Optional[str] = None,
        restart_pipeline: bool = False,
        max_sampler_processes_per_worker: Optional[int] = None,
        collect_valid_results: bool = False,
    ):
        self._initialize_start_train_or_start_test()

        self._collect_valid_results = collect_valid_results

        if not self.disable_config_saving:
            self.save_project_state()

        devices = self.worker_devices(TRAIN_MODE_STR)
        num_workers = len(devices)

        # Be extra careful to ensure that all models start
        # with the same initializations.
        set_seed(self.seed)
        initial_model_state_dict = self.config.create_model(
            sensor_preprocessor_graph=MachineParams.instance_from(
                self.config.machine_params(self.mode)
            ).sensor_preprocessor_graph
        ).state_dict()

        distributed_port = 0 if num_workers == 1 else self.get_port()

        worker_ids = self.worker_ids(TRAIN_MODE_STR)

        model_hash = None
        for trainer_id in worker_ids:
            training_kwargs = dict(
                id=trainer_id,
                checkpoint=checkpoint,
                restart_pipeline=restart_pipeline,
                experiment_name=self.experiment_name,
                config=self.config,
                results_queue=self.queues["results"],
                checkpoints_queue=self.queues["checkpoints"]
                if self.running_validation
                else None,
                checkpoints_dir=self.checkpoint_dir(),
                seed=self.seed,
                deterministic_cudnn=self.deterministic_cudnn,
                mp_ctx=self.mp_ctx,
                num_workers=num_workers,
                device=devices[trainer_id],
                distributed_ip=self.distributed_ip_port.split(":")[0],
                distributed_port=distributed_port,
                max_sampler_processes_per_worker=max_sampler_processes_per_worker,
                initial_model_state_dict=initial_model_state_dict
                if model_hash is None
                else model_hash,
            )
            train: BaseProcess = self.mp_ctx.Process(
                target=self.train_loop,
                kwargs=training_kwargs,
            )
            try:
                train.start()
            except ValueError as e:
                # If the `initial_model_state_dict` is too large we sometimes
                # run into errors passing it with multiprocessing. In such cases
                # we instead hash the state_dict and confirm, in each engine worker, that
                # this hash equals the model the engine worker instantiates.
                if e.args[0] == "too many fds":
                    model_hash = md5_hash_of_state_dict(initial_model_state_dict)
                    training_kwargs["initial_model_state_dict"] = model_hash
                    train = self.mp_ctx.Process(
                        target=self.train_loop,
                        kwargs=training_kwargs,
                    )
                    train.start()
                else:
                    raise e

            self.processes[TRAIN_MODE_STR].append(train)

        get_logger().info(
            "Started {} train processes".format(len(self.processes[TRAIN_MODE_STR]))
        )

        # Validation
        if self.running_validation:
            device = self.worker_devices(VALID_MODE_STR)[0]
            self.init_visualizer(VALID_MODE_STR)
            valid: BaseProcess = self.mp_ctx.Process(
                target=self.valid_loop,
                args=(0,),
                kwargs=dict(
                    config=self.config,
                    results_queue=self.queues["results"],
                    checkpoints_queue=self.queues["checkpoints"],
                    seed=12345,  # TODO allow same order for randomly sampled tasks? Is this any useful anyway?
                    deterministic_cudnn=self.deterministic_cudnn,
                    deterministic_agents=self.deterministic_agents,
                    mp_ctx=self.mp_ctx,
                    device=device,
                    max_sampler_processes_per_worker=max_sampler_processes_per_worker,
                ),
            )
            valid.start()
            self.processes[VALID_MODE_STR].append(valid)

            get_logger().info(
                "Started {} valid processes".format(len(self.processes[VALID_MODE_STR]))
            )
        else:
            get_logger().info(
                "No processes allocated to validation, no validation will be run."
            )

        metrics_file_template: Optional[str] = None

        if self._collect_valid_results:
            metrics_dir = self.metric_path(self.local_start_time_str)
            os.makedirs(metrics_dir, exist_ok=True)
            suffix = "__valid_{}".format(self.local_start_time_str)
            metrics_file_template = os.path.join(
                metrics_dir, "metrics" + suffix + "{:012d}.json"
            )  # template for training steps

            get_logger().info(
                "Saving valid metrics with template {}".format(metrics_file_template)
            )

            # Check output file can be written
            with open(metrics_file_template.format(0), "w") as f:
                json.dump([], f, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

        valid_results = self.log_and_close(
            start_time_str=self.local_start_time_str,
            nworkers=len(worker_ids),  # TODO num_workers once we forward metrics,
            metrics_file=metrics_file_template,
        )

        if not self._collect_valid_results:
            return self.local_start_time_str
        else:
            return self.local_start_time_str, valid_results

    def start_test(
        self,
        checkpoint_path_dir_or_pattern: str,
        approx_ckpt_step_interval: Optional[Union[float, int]] = None,
        max_sampler_processes_per_worker: Optional[int] = None,
        inference_expert: bool = False,
    ) -> List[Dict]:
        self.extra_tag += (
            "__" * (len(self.extra_tag) > 0) + "enforced_test_expert"
        ) * inference_expert
        self._initialize_start_train_or_start_test()

        devices = self.worker_devices(TEST_MODE_STR)
        self.init_visualizer(TEST_MODE_STR)
        num_testers = len(devices)

        distributed_port = 0
        if num_testers > 1:
            distributed_port = find_free_port()

        # TODO Assume tester runs on a single machine
        for tester_it in range(num_testers):
            test: BaseProcess = self.mp_ctx.Process(
                target=self.test_loop,
                args=(tester_it,),
                kwargs=dict(
                    config=self.config,
                    results_queue=self.queues["results"],
                    checkpoints_queue=self.queues["checkpoints"],
                    seed=12345,  # TODO allow same order for randomly sampled tasks? Is this any useful anyway?
                    deterministic_cudnn=self.deterministic_cudnn,
                    deterministic_agents=self.deterministic_agents,
                    mp_ctx=self.mp_ctx,
                    num_workers=num_testers,
                    device=devices[tester_it],
                    max_sampler_processes_per_worker=max_sampler_processes_per_worker,
                    distributed_port=distributed_port,
                    enforce_expert=inference_expert,
                ),
            )

            test.start()
            self.processes[TEST_MODE_STR].append(test)

        get_logger().info(
            "Started {} test processes".format(len(self.processes[TEST_MODE_STR]))
        )

        checkpoint_paths = self.get_checkpoint_files(
            checkpoint_path_dir_or_pattern=checkpoint_path_dir_or_pattern,
            approx_ckpt_step_interval=approx_ckpt_step_interval,
        )
        steps = [self.step_from_checkpoint(cp) for cp in checkpoint_paths]

        get_logger().info("Running test on {} steps {}".format(len(steps), steps))

        for checkpoint_path in checkpoint_paths:
            # Make all testers work on each checkpoint
            for tester_it in range(num_testers):
                self.queues["checkpoints"].put(("eval", checkpoint_path))

        # Signal all testers to terminate cleanly
        for _ in range(num_testers):
            self.queues["checkpoints"].put(("quit", None))

        metrics_dir = self.metric_path(self.local_start_time_str)
        os.makedirs(metrics_dir, exist_ok=True)
        suffix = "__test_{}".format(self.local_start_time_str)
        metrics_file_path = os.path.join(metrics_dir, "metrics" + suffix + ".json")

        get_logger().info("Saving test metrics in {}".format(metrics_file_path))

        # Check output file can be written
        with open(metrics_file_path, "w") as f:
            json.dump([], f, indent=4, sort_keys=True, cls=NumpyJSONEncoder)

        return self.log_and_close(
            start_time_str=self.checkpoint_start_time_str(checkpoint_paths[0]),
            nworkers=num_testers,
            test_steps=steps,
            metrics_file=metrics_file_path,
        )

    @staticmethod
    def checkpoint_start_time_str(checkpoint_file_name):
        parts = checkpoint_file_name.split(os.path.sep)
        assert len(parts) > 1, "{} is not a valid checkpoint path".format(
            checkpoint_file_name
        )
        start_time_str = parts[-2]
        get_logger().info("Using checkpoint start time {}".format(start_time_str))
        return start_time_str

    @property
    def experiment_name(self):
        if len(self.extra_tag) > 0:
            return "{}_{}".format(self.config.tag(), self.extra_tag)
        return self.config.tag()

    def checkpoint_dir(
        self, start_time_str: Optional[str] = None, create_if_none: bool = True
    ):
        folder = os.path.join(
            self.output_dir,
            "checkpoints",
            self.config.tag()
            if self.extra_tag == ""
            else os.path.join(self.config.tag(), self.extra_tag),
            start_time_str or self.local_start_time_str,
        )
        if create_if_none:
            os.makedirs(folder, exist_ok=True)
        return folder

    def log_writer_path(self, start_time_str: str) -> str:
        path = os.path.join(
            self.output_dir,
            "tb",
            self.config.tag()
            if self.extra_tag == ""
            else os.path.join(self.config.tag(), self.extra_tag),
            start_time_str,
        )
        if self.mode == TEST_MODE_STR:
            path = os.path.join(path, "test", self.local_start_time_str)
        return path

    def metric_path(self, start_time_str: str) -> str:
        return os.path.join(
            self.output_dir,
            "metrics",
            self.config.tag()
            if self.extra_tag == ""
            else os.path.join(self.config.tag(), self.extra_tag),
            start_time_str,
        )

    def save_project_state(self):
        base_dir = os.path.join(
            self.output_dir,
            "used_configs",
            self.config.tag()
            if self.extra_tag == ""
            else os.path.join(self.config.tag(), self.extra_tag),
            self.local_start_time_str,
        )
        os.makedirs(base_dir, exist_ok=True)

        # Saving current git diff
        try:
            sha, diff_str = get_git_diff_of_project()
            with open(os.path.join(base_dir, "{}.patch".format(sha)), "w") as f:
                f.write(diff_str)

            get_logger().info("Git diff saved to {}".format(base_dir))
        except subprocess.CalledProcessError:
            get_logger().warning(
                "Failed to get a git diff of the current project."
                f" Is it possible that {os.getcwd()} is not under version control?"
            )

        # Saving configs
        if self.loaded_config_src_files is not None:
            for src_path in self.loaded_config_src_files:
                if src_path == _CONFIG_KWARGS_STR:
                    # We also save key-word arguments passed to to the experiment
                    # initializer.
                    save_path = os.path.join(base_dir, "config_kwargs.json")
                    assert not os.path.exists(
                        save_path
                    ), f"{save_path} should not already exist."
                    with open(save_path, "w") as f:
                        json.dump(json.loads(self.loaded_config_src_files[src_path]), f)
                    continue

                assert os.path.isfile(src_path), "Config file {} not found".format(
                    src_path
                )
                src_path = os.path.abspath(src_path)

                # To prevent overwriting files with the same name, we loop
                # here until we find a prefix (if necessary) to prevent
                # name collisions.
                k = -1
                while True:
                    prefix = "" if k == -1 else "namecollision{}__".format(k)
                    k += 1
                    dst_path = os.path.join(
                        base_dir,
                        "{}{}".format(
                            prefix,
                            os.path.basename(src_path),
                        ),
                    )
                    if not os.path.exists(dst_path):
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        with open(src_path, "r") as f:
                            file_contents = f.read()
                        with open(dst_path, "w") as f:
                            f.write(
                                "### THIS FILE ORIGINALLY LOCATED AT '{}'\n\n{}".format(
                                    src_path, file_contents
                                )
                            )
                        break

        get_logger().info("Config files saved to {}".format(base_dir))

    def process_eval_package(
        self,
        log_writer: Optional[SummaryWriter],
        pkg: LoggingPackage,
        all_results: Optional[List[Any]] = None,
    ):
        training_steps = pkg.training_steps
        checkpoint_file_name = pkg.checkpoint_file_name
        render = pkg.viz_data
        task_outputs = pkg.metric_dicts

        num_tasks = pkg.num_non_empty_metrics_dicts_added
        metric_means = pkg.metrics_tracker.means()

        mode = pkg.mode

        if log_writer is not None:
            log_writer.add_scalar(
                f"{mode}-misc/num_tasks_evaled", num_tasks, training_steps
            )

        message = [f"{mode} {training_steps} steps:"]
        for k in sorted(metric_means.keys()):
            if log_writer is not None:
                log_writer.add_scalar(
                    f"{mode}-metrics/{k}", metric_means[k], training_steps
                )
            message.append(f"{k} {metric_means[k]}")

        if all_results is not None:
            results = copy.deepcopy(metric_means)
            results.update({"training_steps": training_steps, "tasks": task_outputs})
            all_results.append(results)

        message.append(f"tasks {num_tasks} checkpoint {checkpoint_file_name}")
        get_logger().info(" ".join(message))

        if self.visualizer is not None:
            self.visualizer.log(
                log_writer=log_writer,
                task_outputs=task_outputs,
                render=render,
                num_steps=training_steps,
            )

    def process_train_packages(
        self,
        log_writer: Optional[SummaryWriter],
        pkgs: List[LoggingPackage],
        last_steps=0,
        last_offpolicy_steps=0,
        last_time=0.0,
    ):
        assert self.mode == TRAIN_MODE_STR

        current_time = time.time()

        training_steps = pkgs[0].training_steps
        offpolicy_steps = pkgs[0].off_policy_steps
        if log_writer is not None:
            log_writer.add_scalar(
                tag="train-misc/pipeline_stage",
                scalar_value=pkgs[0].pipeline_stage,
                global_step=training_steps,
            )

        def add_prefix(d: Dict[str, Any], tag: str) -> Dict[str, Any]:
            new_dict = {}
            for k, v in d.items():
                if "offpolicy" in k:
                    pass
                elif k.startswith("losses/"):
                    k = f"{self.mode}-{k}"
                else:
                    k = f"{self.mode}-{tag}/{k}"
                new_dict[k] = v
            return new_dict

        metrics_and_train_info_tracker = ScalarMeanTracker()
        for pkg in pkgs:
            metrics_and_train_info_tracker.add_scalars(
                scalars=add_prefix(pkg.metrics_tracker.means(), "metrics"),
                n=add_prefix(pkg.metrics_tracker.counts(), "metrics"),
            )
            metrics_and_train_info_tracker.add_scalars(
                scalars=add_prefix(pkg.train_info_tracker.means(), "misc"),
                n=add_prefix(pkg.train_info_tracker.counts(), "misc"),
            )

        message = [
            "train {} steps {} offpolicy:".format(training_steps, offpolicy_steps)
        ]
        means = metrics_and_train_info_tracker.means()

        for k in sorted(
            means.keys(), key=lambda mean_key: (mean_key.count("/"), mean_key)
        ):
            if log_writer is not None:
                log_writer.add_scalar(k, means[k], training_steps)
            short_key = (
                "/".join(k.split("/")[1:]) if k.startswith("train-") and "/" in k else k
            )
            message.append(f"{short_key} {means[k]:.3g}")
        message += [f"elapsed_time {(current_time - last_time):.3g}s"]

        if last_steps > 0:
            fps = (training_steps - last_steps) / (current_time - last_time)
            message += [f"approx_fps {fps:.3g}"]
            if log_writer is not None:
                log_writer.add_scalar("train-misc/approx_fps", fps, training_steps)

        if last_offpolicy_steps > 0:
            fps = (offpolicy_steps - last_offpolicy_steps) / (current_time - last_time)
            message += ["offpolicy/approx_fps {:.3g}".format(fps)]
            if log_writer is not None:
                log_writer.add_scalar("offpolicy/approx_fps", fps, training_steps)

        get_logger().info(" ".join(message))

        return training_steps, offpolicy_steps, current_time

    def process_test_packages(
        self,
        log_writer: Optional[SummaryWriter],
        pkgs: List[LoggingPackage],
        all_results: Optional[List[Any]] = None,
    ):
        mode = pkgs[0].mode
        assert mode == TEST_MODE_STR

        training_steps = pkgs[0].training_steps

        all_metrics_tracker = ScalarMeanTracker()
        metric_dicts_list, render, checkpoint_file_name = [], {}, []
        for pkg in pkgs:
            all_metrics_tracker.add_scalars(
                scalars=pkg.metrics_tracker.means(), n=pkg.metrics_tracker.counts()
            )
            metric_dicts_list.extend(pkg.metric_dicts)
            if pkg.viz_data is not None:
                render.update(pkg.viz_data)
            checkpoint_file_name.append(pkg.checkpoint_file_name)

        assert all_equal(checkpoint_file_name)

        message = [f"{mode} {training_steps} steps:"]

        metric_means = all_metrics_tracker.means()
        for k in sorted(metric_means.keys()):
            if log_writer is not None:
                log_writer.add_scalar(
                    f"{mode}-metrics/{k}", metric_means[k], training_steps
                )
            message.append(k + " {:.3g}".format(metric_means[k]))

        if all_results is not None:
            results = copy.deepcopy(metric_means)
            results.update(
                {"training_steps": training_steps, "tasks": metric_dicts_list}
            )
            all_results.append(results)

        num_tasks = sum([pkg.num_non_empty_metrics_dicts_added for pkg in pkgs])
        if log_writer is not None:
            log_writer.add_scalar(
                f"{mode}-misc/num_tasks_evaled", num_tasks, training_steps
            )

        message.append(
            "tasks {} checkpoint {}".format(num_tasks, checkpoint_file_name[0])
        )
        get_logger().info(" ".join(message))

        if self.visualizer is not None:
            self.visualizer.log(
                log_writer=log_writer,
                task_outputs=metric_dicts_list,
                render=render,
                num_steps=training_steps,
            )

    def log_and_close(
        self,
        start_time_str: str,
        nworkers: int,
        test_steps: Sequence[int] = (),
        metrics_file: Optional[str] = None,
    ) -> List[Dict]:
        finalized = False

        log_writer: Optional[SummaryWriter] = None
        if not self.disable_tensorboard:
            log_writer = SummaryWriter(
                log_dir=self.log_writer_path(start_time_str),
                filename_suffix="__{}_{}".format(self.mode, self.local_start_time_str),
            )

        # To aggregate/buffer metrics from trainers/testers
        collected: List[LoggingPackage] = []
        last_train_steps = 0
        last_offpolicy_steps = 0
        last_train_time = time.time()
        # test_steps = sorted(test_steps, reverse=True)
        eval_results: List[Dict] = []
        unfinished_workers = nworkers

        try:
            while True:
                try:
                    package: Union[
                        LoggingPackage, Union[Tuple[str, Any], Tuple[str, Any, Any]]
                    ] = self.queues["results"].get(timeout=1)

                    if isinstance(package, LoggingPackage):
                        pkg_mode = package.mode

                        if pkg_mode == TRAIN_MODE_STR:
                            collected.append(package)
                            if len(collected) >= nworkers:

                                collected = sorted(
                                    collected,
                                    key=lambda pkg: (
                                        pkg.training_steps,
                                        pkg.off_policy_steps,
                                    ),
                                )

                                if (
                                    collected[nworkers - 1].training_steps
                                    == collected[0].training_steps
                                    and collected[nworkers - 1].off_policy_steps
                                    == collected[0].off_policy_steps
                                ):  # ensure nworkers have provided the same num_steps
                                    (
                                        last_train_steps,
                                        last_offpolicy_steps,
                                        last_train_time,
                                    ) = self.process_train_packages(
                                        log_writer=log_writer,
                                        pkgs=collected[:nworkers],
                                        last_steps=last_train_steps,
                                        last_offpolicy_steps=last_offpolicy_steps,
                                        last_time=last_train_time,
                                    )
                                    collected = collected[nworkers:]
                                elif len(collected) > 2 * nworkers:
                                    get_logger().warning(
                                        "Unable to aggregate train packages from all {} workers"
                                        "after {} packages collected".format(
                                            nworkers, len(collected)
                                        )
                                    )
                        elif pkg_mode == VALID_MODE_STR:  # they all come from a single worker
                            if (
                                package.training_steps is not None
                            ):  # no validation samplers
                                self.process_eval_package(
                                    log_writer=log_writer,
                                    pkg=package,
                                    all_results=eval_results
                                    if self._collect_valid_results
                                    else None,
                                )

                                if metrics_file is not None:
                                    with open(
                                        metrics_file.format(package.training_steps), "w"
                                    ) as f:
                                        json.dump(
                                            eval_results[-1],
                                            f,
                                            indent=4,
                                            sort_keys=True,
                                            cls=NumpyJSONEncoder,
                                        )
                                        get_logger().info(
                                            "Written valid results file {}".format(
                                                metrics_file.format(
                                                    package.training_steps
                                                ),
                                            )
                                        )

                            if (
                                finalized and self.queues["checkpoints"].empty()
                            ):  # assume queue is actually empty after trainer finished and no checkpoints in queue
                                break
                        elif pkg_mode == TEST_MODE_STR:
                            collected.append(package)
                            if len(collected) >= nworkers:
                                collected = sorted(
                                    collected, key=lambda x: x.training_steps
                                )  # sort by num_steps
                                if (
                                    collected[nworkers - 1].training_steps
                                    == collected[0].training_steps
                                ):  # ensure nworkers have provided the same num_steps
                                    self.process_test_packages(
                                        log_writer=log_writer,
                                        pkgs=collected[:nworkers],
                                        all_results=eval_results,
                                    )

                                    collected = collected[nworkers:]
                                    with open(metrics_file, "w") as f:
                                        json.dump(
                                            eval_results,
                                            f,
                                            indent=4,
                                            sort_keys=True,
                                            cls=NumpyJSONEncoder,
                                        )
                                        get_logger().info(
                                            "Updated {} up to checkpoint {}".format(
                                                metrics_file,
                                                test_steps[len(eval_results) - 1],
                                            )
                                        )
                        else:
                            get_logger().error(
                                f"Runner received unknown package of type {pkg_mode}"
                            )
                    else:
                        pkg_mode = package[0]

                        if pkg_mode == "train_stopped":
                            if package[1] == 0:
                                finalized = True
                                if not self.running_validation:
                                    get_logger().info(
                                        "Terminating runner after trainer done (no validation)"
                                    )
                                    break
                            else:
                                raise Exception(
                                    "Train worker {} abnormally terminated".format(
                                        package[1] - 1
                                    )
                                )
                        elif pkg_mode == "valid_stopped":
                            raise Exception(
                                "Valid worker {} abnormally terminated".format(
                                    package[1] - 1
                                )
                            )
                        elif pkg_mode == "test_stopped":
                            if package[1] == 0:
                                unfinished_workers -= 1
                                if unfinished_workers == 0:
                                    get_logger().info(
                                        "Last tester finished. Terminating"
                                    )
                                    finalized = True
                                    break
                            else:
                                raise RuntimeError(
                                    "Test worker {} abnormally terminated".format(
                                        package[1] - 1
                                    )
                                )
                        else:
                            get_logger().error(
                                f"Runner received invalid package tuple {package}"
                            )
                except queue.Empty as _:
                    if all(
                        p.exitcode is not None
                        for p in itertools.chain(*self.processes.values())
                    ):
                        break
        except KeyboardInterrupt:
            get_logger().info("KeyboardInterrupt. Terminating runner.")
        except Exception:
            get_logger().error("Encountered Exception. Terminating runner.")
            get_logger().exception(traceback.format_exc())
        finally:
            if finalized:
                get_logger().info("Done")
            if log_writer is not None:
                log_writer.close()
            self.close()
            return eval_results

    def get_checkpoint_files(
        self,
        checkpoint_path_dir_or_pattern: str,
        approx_ckpt_step_interval: Optional[int] = None,
    ):

        if os.path.isdir(checkpoint_path_dir_or_pattern):
            # The fragment is a path to a directory, lets use this directory
            # as the base dir to search for checkpoints
            checkpoint_path_dir_or_pattern = os.path.join(
                checkpoint_path_dir_or_pattern, "*.pt"
            )

        ckpt_paths = glob.glob(checkpoint_path_dir_or_pattern, recursive=True)

        if len(ckpt_paths) == 0:
            raise FileNotFoundError(
                f"Could not find any checkpoints at {os.path.abspath(checkpoint_path_dir_or_pattern)}, is it possible"
                f" the path has been mispecified?"
            )

        step_count_ckpt_pairs = [(self.step_from_checkpoint(p), p) for p in ckpt_paths]
        step_count_ckpt_pairs.sort()
        ckpts_paths = [p for _, p in step_count_ckpt_pairs]
        step_counts = np.array([sc for sc, _ in step_count_ckpt_pairs])

        if approx_ckpt_step_interval is not None:
            assert (
                approx_ckpt_step_interval > 0
            ), "`approx_ckpt_step_interval` must be >0"
            inds_to_eval = set()
            for i in range(
                math.ceil(step_count_ckpt_pairs[-1][0] / approx_ckpt_step_interval) + 1
            ):
                inds_to_eval.add(
                    int(np.argmin(np.abs(step_counts - i * approx_ckpt_step_interval)))
                )

            ckpts_paths = [ckpts_paths[ind] for ind in sorted(list(inds_to_eval))]
        return ckpts_paths

    @staticmethod
    def step_from_checkpoint(ckpt_path: str) -> int:
        parts = os.path.basename(ckpt_path).split("__")
        for part in parts:
            if "steps_" in part:
                possible_num = part.split("_")[-1].split(".")[0]
                if possible_num.isdigit():
                    return int(possible_num)

        get_logger().warning(
            f"The checkpoint {os.path.basename(ckpt_path)} does not follow the checkpoint naming convention"
            f" used by AllenAct. As a fall back we must load the checkpoint into memory to find the"
            f" training step count, this may increase startup time if the checkpoints are large or many"
            f" must be loaded in sequence."
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return ckpt["total_steps"]

    def close(self, verbose=True):
        if self._is_closed:
            return

        def logif(s: Union[str, Exception]):
            if verbose:
                if isinstance(s, str):
                    get_logger().info(s)
                elif isinstance(s, Exception):
                    get_logger().exception(traceback.format_exc())
                else:
                    raise NotImplementedError()

        # First send termination signals
        for process_type in self.processes:
            for it, process in enumerate(self.processes[process_type]):
                if process.is_alive():
                    logif("Terminating {} {}".format(process_type, it))
                    process.terminate()

        # Now join processes
        for process_type in self.processes:
            for it, process in enumerate(self.processes[process_type]):
                try:
                    logif("Joining {} {}".format(process_type, it))
                    process.join(1)
                    logif("Closed {} {}".format(process_type, it))
                except Exception as e:
                    logif(
                        "Exception raised when closing {} {}".format(process_type, it)
                    )
                    logif(e)

        self.processes.clear()
        self._is_closed = True

    def __del__(self):
        self.close(verbose=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(verbose=True)
