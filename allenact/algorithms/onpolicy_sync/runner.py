"""Defines the reinforcement learning `OnPolicyRunner`."""
import copy
import glob
import itertools
import json
import os
import queue
import signal
import subprocess
import time
import traceback
from collections import defaultdict
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Optional, Dict, Union, Tuple, Sequence, List, Any

import torch.multiprocessing as mp
from setproctitle import setproctitle as ptitle

from allenact.algorithms.onpolicy_sync.engine import (
    OnPolicyTrainer,
    OnPolicyInference,
)
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.utils.experiment_utils import (
    ScalarMeanTracker,
    set_deterministic_cudnn,
    set_seed,
    LoggingPackage,
)
from allenact.utils.misc_utils import all_equal, get_git_diff_of_project
from allenact.utils.system import get_logger, find_free_port
from allenact.utils.tensor_utils import SummaryWriter

# Has results queue (aggregated per trainer), checkpoints queue and mp context
# Instantiates train, validate, and test workers
# Logging
# Saves configs, makes folder for trainer models
from allenact.utils.viz_utils import VizSuite


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
        multiprocessing_start_method: str = "forkserver",
        extra_tag: str = "",
        disable_tensorboard: bool = False,
        disable_config_saving: bool = False,
    ):
        self.config = config
        self.output_dir = output_dir
        self.loaded_config_src_files = loaded_config_src_files
        self.seed = seed
        self.deterministic_cudnn = deterministic_cudnn
        self.mp_ctx = self.init_context(mp_ctx, multiprocessing_start_method)
        self.extra_tag = extra_tag
        self.mode = mode
        self.visualizer: Optional[VizSuite] = None
        self.deterministic_agents = deterministic_agents
        self.disable_tensorboard = disable_tensorboard
        self.disable_config_saving = disable_config_saving

        assert self.mode in [
            "train",
            "test",
        ], "Only 'train' and 'test' modes supported in runner"

        if self.deterministic_cudnn:
            set_deterministic_cudnn()

        if self.seed is not None:
            set_seed(self.seed)

        self.queues = {
            "results": self.mp_ctx.Queue(),
            "checkpoints": self.mp_ctx.Queue(),
        }

        self.processes: Dict[str, List[Union[BaseProcess, mp.Process]]] = defaultdict(
            list
        )

        self.current_checkpoint = None

        self.local_start_time_str = time.strftime(
            "%Y-%m-%d_%H-%M-%S", time.localtime(time.time())
        )

        self._is_closed: bool = False

    @property
    def running_validation(self):
        return (
            sum(
                MachineParams.instance_from(
                    self.config.machine_params("valid")
                ).nprocesses
            )
            > 0
        )

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

    def init_visualizer(self, mode: str):
        if not self.disable_tensorboard:
            # Note: Avoid instantiating anything in machine_params (use Builder if needed)
            machine_params = MachineParams.instance_from(
                self.config.machine_params(mode)
            )
            self.visualizer = machine_params.visualizer

    @staticmethod
    def init_process(mode: str, id: int):
        ptitle("{}-{}".format(mode, id))

        def sigterm_handler(_signo, _stack_frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, sigterm_handler)

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
        OnPolicyRunner.init_process("Train", id)
        engine_kwargs["mode"] = "train"
        engine_kwargs["worker_id"] = id
        get_logger().info("train {} args {}".format(id, engine_kwargs))

        trainer: OnPolicyTrainer = OnPolicyRunner.init_worker(
            engine_class=OnPolicyTrainer, args=engine_args, kwargs=engine_kwargs
        )
        if trainer is not None:
            trainer.train(
                checkpoint_file_name=checkpoint, restart_pipeline=restart_pipeline
            )

    @staticmethod
    def valid_loop(id: int = 0, *engine_args, **engine_kwargs):
        OnPolicyRunner.init_process("Valid", id)
        engine_kwargs["mode"] = "valid"
        engine_kwargs["worker_id"] = id
        get_logger().info("valid {} args {}".format(id, engine_kwargs))

        valid = OnPolicyRunner.init_worker(
            engine_class=OnPolicyInference, args=engine_args, kwargs=engine_kwargs
        )
        if valid is not None:
            valid.process_checkpoints()  # gets checkpoints via queue

    @staticmethod
    def test_loop(id: int = 0, *engine_args, **engine_kwargs):
        OnPolicyRunner.init_process("Test", id)
        engine_kwargs["mode"] = "test"
        engine_kwargs["worker_id"] = id
        get_logger().info("test {} args {}".format(id, engine_kwargs))

        test = OnPolicyRunner.init_worker(OnPolicyInference, engine_args, engine_kwargs)
        if test is not None:
            test.process_checkpoints()  # gets checkpoints via queue

    def start_train(
        self,
        checkpoint: Optional[str] = None,
        restart_pipeline: bool = False,
        max_sampler_processes_per_worker: Optional[int] = None,
    ):
        if not self.disable_config_saving:
            self.save_project_state()

        devices = self.worker_devices("train")
        num_workers = len(devices)

        seed = (
            self.seed
        )  # same for all workers. used during initialization of the model

        distributed_port = 0
        if num_workers > 1:
            distributed_port = find_free_port()

        for trainer_it in range(num_workers):
            train: BaseProcess = self.mp_ctx.Process(
                target=self.train_loop,
                kwargs=dict(
                    id=trainer_it,
                    checkpoint=checkpoint,
                    restart_pipeline=restart_pipeline,
                    experiment_name=self.experiment_name,
                    config=self.config,
                    results_queue=self.queues["results"],
                    checkpoints_queue=self.queues["checkpoints"]
                    if self.running_validation
                    else None,
                    checkpoints_dir=self.checkpoint_dir(),
                    seed=seed,
                    deterministic_cudnn=self.deterministic_cudnn,
                    mp_ctx=self.mp_ctx,
                    num_workers=num_workers,
                    device=devices[trainer_it],
                    distributed_port=distributed_port,
                    max_sampler_processes_per_worker=max_sampler_processes_per_worker,
                ),
            )
            train.start()
            self.processes["train"].append(train)

        get_logger().info(
            "Started {} train processes".format(len(self.processes["train"]))
        )

        # Validation
        if self.running_validation:
            device = self.worker_devices("valid")[0]
            self.init_visualizer("valid")
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
            self.processes["valid"].append(valid)

            get_logger().info(
                "Started {} valid processes".format(len(self.processes["valid"]))
            )
        else:
            get_logger().info(
                "No processes allocated to validation, no validation will be run."
            )

        self.log(self.local_start_time_str, num_workers)

        return self.local_start_time_str

    def start_test(
        self,
        experiment_date: str,
        checkpoint: Optional[str] = None,
        skip_checkpoints: int = 0,
        max_sampler_processes_per_worker: Optional[int] = None,
    ):
        devices = self.worker_devices("test")
        self.init_visualizer("test")
        num_testers = len(devices)

        distributed_port = 0
        if num_testers > 1:
            distributed_port = find_free_port()

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
                ),
            )

            test.start()
            self.processes["test"].append(test)

        get_logger().info(
            "Started {} test processes".format(len(self.processes["test"]))
        )

        checkpoints = self.get_checkpoint_files(
            experiment_date, checkpoint, skip_checkpoints
        )
        steps = [self.step_from_checkpoint(cp) for cp in checkpoints]

        get_logger().info("Running test on {} steps {}".format(len(steps), steps))

        for checkpoint in checkpoints:
            # Make all testers work on each checkpoint
            for tester_it in range(num_testers):
                self.queues["checkpoints"].put(("eval", checkpoint))
        # Signal all testers to terminate cleanly
        for _ in range(num_testers):
            self.queues["checkpoints"].put(("quit", None))

        metric_folder = self.metric_path(experiment_date)
        os.makedirs(metric_folder, exist_ok=True)
        suffix = "__test_{}".format(self.local_start_time_str)
        fname = os.path.join(metric_folder, "metrics" + suffix + ".json")

        get_logger().info("Saving metrics in {}".format(fname))

        # Check output file can be written
        with open(fname, "w") as f:
            json.dump([], f, indent=4, sort_keys=True)

        return self.log(
            self.checkpoint_start_time_str(checkpoints[0]), num_testers, steps, fname
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

    def checkpoint_dir(self, start_time_str=None):
        folder = os.path.join(
            self.output_dir,
            "checkpoints",
            self.config.tag()
            if self.extra_tag == ""
            else os.path.join(self.config.tag(), self.extra_tag),
            start_time_str or self.local_start_time_str,
        )
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
        if self.mode == "test":
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
                        base_dir, "{}{}".format(prefix, os.path.basename(src_path),),
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
        self, log_writer: Optional[SummaryWriter], pkg: LoggingPackage
    ):
        training_steps = pkg.training_steps
        checkpoint_file_name = pkg.checkpoint_file_name
        render = pkg.viz_data
        task_outputs = pkg.metric_dicts

        num_tasks = pkg.num_non_empty_metrics_dicts_added
        metric_means = pkg.metrics_tracker.means()

        mode = pkg.mode

        if log_writer is not None:
            log_writer.add_scalar(f"{mode}/num_tasks_evaled", num_tasks, training_steps)

        message = [f"{mode} {training_steps} steps:"]
        for k in sorted(metric_means.keys()):
            if log_writer is not None:
                log_writer.add_scalar(f"{mode}/{k}", metric_means[k], training_steps)
            message.append(f"{k} {metric_means[k]}")
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
        assert self.mode == "train"

        current_time = time.time()

        training_steps = pkgs[0].training_steps
        offpolicy_steps = pkgs[0].off_policy_steps
        if log_writer is not None:
            log_writer.add_scalar(
                tag="train/pipeline_stage",
                scalar_value=pkgs[0].pipeline_stage,
                global_step=training_steps,
            )

        metrics_and_train_info_tracker = ScalarMeanTracker()
        for pkg in pkgs:
            metrics_and_train_info_tracker.add_scalars(
                scalars=pkg.metrics_tracker.means(), n=pkg.metrics_tracker.counts()
            )
            metrics_and_train_info_tracker.add_scalars(
                scalars=pkg.train_info_tracker.means(),
                n=pkg.train_info_tracker.counts(),
            )

        message = [
            "train {} steps {} offpolicy:".format(training_steps, offpolicy_steps)
        ]
        means = metrics_and_train_info_tracker.means()
        for k in sorted(means.keys(), key=lambda mean_key: ("/" in mean_key, mean_key)):
            if log_writer is not None:
                if "offpolicy" not in k:
                    log_writer.add_scalar(
                        "{}/".format(self.mode) + k, means[k], training_steps
                    )
                else:
                    log_writer.add_scalar(k, means[k], training_steps)
            message.append(k + " {:.3g}".format(means[k]))
        message += ["elapsed_time {:.3g}s".format(current_time - last_time)]

        if last_steps > 0:
            fps = (training_steps - last_steps) / (current_time - last_time)
            message += ["approx_fps {:.3g}".format(fps)]
            if log_writer is not None:
                log_writer.add_scalar("train/approx_fps", fps, training_steps)

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
        assert mode == "test"

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
                log_writer.add_scalar(f"{mode}/{k}", metric_means[k], training_steps)
            message.append(k + " {:.3g}".format(metric_means[k]))

        if all_results is not None:
            results = copy.deepcopy(metric_means)
            results.update(
                {"training_steps": training_steps, "tasks": metric_dicts_list}
            )
            all_results.append(results)

        num_tasks = sum([pkg.num_non_empty_metrics_dicts_added for pkg in pkgs])
        if log_writer is not None:
            log_writer.add_scalar(f"{mode}/num_tasks_evaled", num_tasks, training_steps)

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

    def log(
        self,
        start_time_str: str,
        nworkers: int,
        test_steps: Sequence[int] = (),
        metrics_file: Optional[str] = None,
    ):
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
        test_results: List[Dict] = []
        unfinished_workers = nworkers

        try:
            while True:
                try:
                    package: Union[
                        LoggingPackage, Union[Tuple[str, Any], Tuple[str, Any, Any]]
                    ] = self.queues["results"].get(timeout=1)

                    if isinstance(package, LoggingPackage):
                        pkg_mode = package.mode

                        if pkg_mode == "train":
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
                        elif pkg_mode == "valid":  # they all come from a single worker
                            if (
                                package.training_steps is not None
                            ):  # no validation samplers
                                self.process_eval_package(
                                    log_writer=log_writer, pkg=package
                                )
                            if (
                                finalized and self.queues["checkpoints"].empty()
                            ):  # assume queue is actually empty after trainer finished and no checkpoints in queue
                                break
                        elif pkg_mode == "test":
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
                                        all_results=test_results,
                                    )
                                    collected = collected[nworkers:]
                                    with open(metrics_file, "w") as f:
                                        json.dump(
                                            test_results, f, indent=4, sort_keys=True
                                        )
                                        get_logger().debug(
                                            "Updated {} up to checkpoint {}".format(
                                                metrics_file,
                                                test_steps[len(test_results) - 1],
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
            return test_results

    def get_checkpoint_files(
        self,
        experiment_date: str,
        checkpoint_file_name: Optional[str] = None,
        skip_checkpoints: int = 0,
    ):
        if checkpoint_file_name is not None:
            return [checkpoint_file_name]
        files = glob.glob(
            os.path.join(self.checkpoint_dir(experiment_date), "exp_*.pt")
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

    @staticmethod
    def step_from_checkpoint(name: str):
        parts = name.split("__")
        for part in parts:
            if "steps_" in part:
                return int(part.split("_")[-1].split(".")[0])
        return -1

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

        for process_type in self.processes:
            for it, process in enumerate(self.processes[process_type]):
                try:
                    if process.is_alive():
                        logif("Closing {} {}".format(process_type, it))
                        process.terminate()
                    logif("Joining {} {}".format(process_type, it))
                    process.join(1)
                    logif("Closed {} {}".format(process_type, it))
                except Exception as e:
                    logif(
                        "Exception raised when closing {} {}".format(process_type, it)
                    )
                    logif(e)

        self._is_closed = True

    def __del__(self):
        self.close(verbose=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(verbose=True)
