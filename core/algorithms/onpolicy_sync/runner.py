"""Defines the reinforcement learning `OnPolicyRunner`."""
import copy
import glob
import itertools
import json
import os
import queue
import shutil
import signal
import time
import traceback
from collections import OrderedDict, defaultdict
from multiprocessing.context import BaseContext
from typing import Optional, Dict, Union, Tuple, Sequence, List, Any

import torch
import torch.distributions
import torch.multiprocessing as mp
import torch.optim
from setproctitle import setproctitle as ptitle

from core.algorithms.onpolicy_sync.engine import (
    OnPolicyTrainer,
    OnPolicyInference,
)
from core.base_abstractions.experiment_config import ExperimentConfig
from utils.experiment_utils import ScalarMeanTracker, set_deterministic_cudnn, set_seed
from utils.misc_utils import all_equal, get_git_diff_of_project
from utils.system import get_logger, find_free_port
from utils.tensor_utils import SummaryWriter

# Has results queue (aggregated per trainer), checkpoints queue and mp context
# Instantiates train, validate, and test workers
# Logging
# Saves configs, makes folder for trainer models
from utils.viz_utils import SimpleViz


class OnPolicyRunner(object):
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str,
        loaded_config_src_files: Optional[Dict[str, Tuple[str, str]]],
        seed: Optional[int] = None,
        mode: str = "train",
        deterministic_cudnn: bool = False,
        mp_ctx: Optional[BaseContext] = None,
        multiprocessing_start_method: str = "forkserver",
        extra_tag: str = "",
    ):
        self.config = config
        self.output_dir = output_dir
        self.loaded_config_src_files = loaded_config_src_files
        self.seed = seed
        self.deterministic_cudnn = deterministic_cudnn
        self.mp_ctx = self.init_context(mp_ctx, multiprocessing_start_method)
        self.extra_tag = extra_tag
        self.mode = mode
        self.visualizer: Optional[SimpleViz] = None

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

        self.processes: Dict[str, List[mp.Process]] = defaultdict(list)

        self.current_checkpoint = None

        self.local_start_time_str = time.strftime(
            "%Y-%m-%d_%H-%M-%S", time.localtime(time.time())
        )

        self.scalars = ScalarMeanTracker()

        self._is_closed: bool = False

    @property
    def running_validation(self):
        return self.config.machine_params("valid")["nprocesses"] > 0

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
        # Note: Avoid instantiating preprocessors in machine_params (use Builder if needed)
        devices = self.config.machine_params(mode)["gpu_ids"]
        if len(devices) > 0:
            if torch.device(devices[0]) == torch.device("cpu"):
                assert all_equal(
                    devices
                ), "Specified devices {} must be all non-negative integers or all equal to 'cpu'".format(
                    devices
                )
                devices = [torch.device(d) for d in devices]
            else:
                assert all(
                    [gpu_id >= 0 for gpu_id in devices]
                ), "all gpu_ids must be >= 0"
                assert torch.cuda.device_count() > max(
                    set(devices)
                ), "{} CUDA devices available for requested {} gpu ids {}".format(
                    torch.cuda.device_count(), mode, devices
                )
        else:
            devices = [torch.device("cpu")]
        get_logger().info(
            "Using {} {} workers on devices {}".format(len(devices), mode, devices)
        )
        return devices

    def get_visualizer(self, mode: str):
        # Note: Avoid instantiating preprocessors in machine_params (use Builder if needed)
        params = self.config.machine_params(mode)
        if "visualizer" in params and params["visualizer"] is not None:
            self.visualizer = params["visualizer"]()  # it's a Builder!

    @staticmethod
    def init_process(mode, id):
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
        **engine_kwargs
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
        self.save_project_state()

        devices = self.worker_devices("train")
        num_trainers = len(devices)

        seed = (
            self.seed
        )  # same for all workers. used during initialization of the model

        distributed_port = 0
        distributed_barrier = None
        if num_trainers > 1:
            distributed_port = find_free_port()
            distributed_barrier = self.mp_ctx.Barrier(num_trainers)

        for trainer_it in range(num_trainers):
            train: mp.process.BaseProcess = self.mp_ctx.Process(
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
                    num_workers=num_trainers,
                    device=devices[trainer_it],
                    distributed_port=distributed_port,
                    distributed_barrier=distributed_barrier,
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
            self.get_visualizer("valid")
            valid: mp.process.BaseProcess = self.mp_ctx.Process(
                target=self.valid_loop,
                args=(0,),
                kwargs=dict(
                    config=self.config,
                    results_queue=self.queues["results"],
                    checkpoints_queue=self.queues["checkpoints"],
                    seed=12345,  # TODO allow same order for randomly sampled tasks? Is this any useful anyway?
                    deterministic_cudnn=self.deterministic_cudnn,
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

        self.log(self.local_start_time_str, num_trainers)

        return self.local_start_time_str

    def start_test(
        self,
        experiment_date: str,
        cp: Optional[str] = None,
        skip_checkpoints: int = 0,
        max_sampler_processes_per_worker: Optional[int] = None,
    ):
        devices = self.worker_devices("test")
        self.get_visualizer("test")
        num_testers = len(devices)

        distributed_barrier = None
        if num_testers > 1:
            distributed_barrier = self.mp_ctx.Barrier(num_testers)

        for tester_it in range(num_testers):
            test: mp.process.BaseProcess = self.mp_ctx.Process(
                target=self.test_loop,
                args=(tester_it,),
                kwargs=dict(
                    config=self.config,
                    results_queue=self.queues["results"],
                    checkpoints_queue=self.queues["checkpoints"],
                    seed=12345,  # TODO allow same order for randomly sampled tasks? Is this any useful anyway?
                    deterministic_cudnn=self.deterministic_cudnn,
                    mp_ctx=self.mp_ctx,
                    num_workers=num_testers,
                    device=devices[tester_it],
                    distributed_barrier=distributed_barrier,
                    max_sampler_processes_per_worker=max_sampler_processes_per_worker,
                ),
            )

            test.start()
            self.processes["test"].append(test)

        get_logger().info(
            "Started {} test processes".format(len(self.processes["test"]))
        )

        checkpoints = self.get_checkpoint_files(experiment_date, cp, skip_checkpoints)
        steps = [self.step_from_checkpoint(cp) for cp in checkpoints]

        get_logger().info("Running test on {} steps {}".format(len(steps), steps))

        for cp in checkpoints:
            # Make all testers work on each checkpoint
            for tester_it in range(num_testers):
                self.queues["checkpoints"].put(("eval", cp))
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

    def log_writer_path(self, start_time_str) -> str:
        return os.path.join(
            self.output_dir,
            "tb",
            self.config.tag()
            if self.extra_tag == ""
            else os.path.join(self.config.tag(), self.extra_tag),
            start_time_str,
        )

    def metric_path(self, start_time_str) -> str:
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
        sha, diff_str = get_git_diff_of_project()
        with open(os.path.join(base_dir, "{}.patch".format(sha)), "w") as f:
            f.write(diff_str)

        get_logger().info("Git diff saved to {}".format(base_dir))

        # Recursively saving configs
        if self.loaded_config_src_files is not None:
            for file in self.loaded_config_src_files:
                base, module = self.loaded_config_src_files[file]
                parts = module.split(".")

                src_file = os.path.sep.join([base] + parts) + ".py"
                assert os.path.isfile(src_file), "Config file {} not found".format(
                    src_file
                )

                dst_file = os.path.join(base_dir, os.path.join(*parts[1:]),) + ".py"
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy(src_file, dst_file)

        get_logger().info("Config files saved to {}".format(base_dir))

    def process_eval_package(
        self, log_writer, pkg, all_results: Optional[List[Any]] = None
    ):
        pkg_type, payload, steps = pkg
        metrics_pkg, task_outputs, render, checkpoint_file_name = payload

        metrics_type, metrics_payload, num_tasks = metrics_pkg

        mode = pkg_type.split("_")[0]

        metrics = OrderedDict(
            sorted(
                [(k, v) for k, v in metrics_payload.items() if k != "task_info"],
                key=lambda x: x[0],
            )
        )

        if all_results is not None:
            results = copy.deepcopy(metrics)
            results.update({"training_steps": steps, "tasks": task_outputs})
            all_results.append(results)

        message = ["{} {} steps:".format(mode, steps)]
        for k in metrics:
            log_writer.add_scalar("{}/".format(mode) + k, metrics[k], steps)
            message.append(k + " {}".format(metrics[k]))
        message.append("tasks {} checkpoint {}".format(num_tasks, checkpoint_file_name))
        get_logger().info(" ".join(message))

        # if render is not None:
        #     log_writer.add_vid("{}/agent_view".format(mode), render, steps)

        if self.visualizer is not None:
            self.visualizer.log(log_writer, task_outputs, render, steps)

    def aggregate_infos(self, log_writer, infos, steps, return_metrics=False):
        nsamples = sum(info[2] for info in infos)
        valid_infos = sum(info[2] > 0 for info in infos)

        # assert nsamples != 0, "Attempting to aggregate infos with 0 samples".format(type)
        assert (
            self.scalars.empty
        ), "Attempting to aggregate with non-empty ScalarMeanTracker"

        for name, payload, nsamps in infos:
            assert nsamps >= 0, "negative ({}) samples in info".format(nsamps)
            if nsamps > 0:
                self.scalars.add_scalars(
                    {
                        k: valid_infos * payload[k] * nsamps / nsamples for k in payload
                    }  # pop divides by valid_infos
                )

        message = []
        metrics = None
        if nsamples > 0:
            summary = self.scalars.pop_and_reset()

            metrics = OrderedDict(
                sorted(
                    [(k, v) for k, v in summary.items() if k != "task_info"],
                    key=lambda x: x[0],
                )
            )

            for k in metrics:
                log_writer.add_scalar("{}/".format(self.mode) + k, metrics[k], steps)
                message.append(k + " {:.3g}".format(metrics[k]))

        if not return_metrics:
            return message
        else:
            return message, metrics

    def process_train_packages(self, log_writer, pkgs, last_steps=0, last_time=0.0):
        current_time = time.time()

        pkg_types, payloads, all_steps = [vals for vals in zip(*pkgs)]

        steps = all_steps[0]

        all_info_types = [worker_pkgs for worker_pkgs in zip(*payloads)]

        message = ["train {} steps:".format(steps)]
        for info_type in all_info_types:
            message += self.aggregate_infos(log_writer, info_type, steps)
        message += ["elapsed_time {:.3g}s".format(current_time - last_time)]

        if last_steps > 0:
            fps = (steps - last_steps) / (current_time - last_time)
            message += ["approx_fps {:.3g}".format(fps)]
            log_writer.add_scalar("train/approx_fps", fps, steps)
        get_logger().info(" ".join(message))

        return steps, current_time

    def process_test_packages(
        self, log_writer, pkgs, all_results: Optional[List[Any]] = None
    ):
        pkg_types, payloads, all_steps = [vals for vals in zip(*pkgs)]
        steps = all_steps[0]
        metrics_pkg, task_outputs, render, checkpoint_file_name = [], [], {}, []
        for payload in payloads:
            mpkg, touts, rndr, cpfname = payload
            metrics_pkg.append(mpkg)
            task_outputs.extend(touts)
            if rndr is not None:
                render.update(rndr)
            checkpoint_file_name.append(cpfname)

        mode = pkg_types[0].split("_")[0]

        message = ["{} {} steps:".format(mode, steps)]
        # for k in metrics:
        #     log_writer.add_scalar("{}/".format(mode) + k, metrics[k], steps)
        #     message.append(k + " {}".format(metrics[k]))
        msg, mets = self.aggregate_infos(
            log_writer, metrics_pkg, steps, return_metrics=all_results is not None
        )
        message += msg
        if all_results is not None:
            results = copy.deepcopy(mets)
            results.update({"training_steps": steps, "tasks": task_outputs})
            all_results.append(results)

        num_tasks = sum([mpkg[2] for mpkg in metrics_pkg])
        message.append(
            "tasks {} checkpoint {}".format(num_tasks, checkpoint_file_name[0])
        )
        get_logger().info(" ".join(message))

        # if render is not None:
        #     log_writer.add_vid("{}/agent_view".format(mode), render, steps)

        if self.visualizer is not None:
            self.visualizer.log(log_writer, task_outputs, render, steps)

    def log(
        self,
        start_time_str: str,
        nworkers: int,
        test_steps: Sequence[int] = (),
        metrics_file: Optional[str] = None,
    ):
        finalized = False

        log_writer = SummaryWriter(
            log_dir=self.log_writer_path(start_time_str),
            filename_suffix="__{}_{}".format(self.mode, self.local_start_time_str),
        )

        # To aggregate/buffer metrics from trainers/testers
        collected = []
        last_train_steps = 0
        last_train_time = time.time()
        # test_steps = sorted(test_steps, reverse=True)
        test_results: List[Dict] = []
        unfinished_workers = nworkers

        try:
            while True:
                try:
                    package = self.queues["results"].get(timeout=1)
                    if package[0] == "train_package":
                        collected.append(package)
                        if len(collected) >= nworkers:
                            collected = sorted(
                                collected, key=lambda x: x[2]
                            )  # sort by num_steps
                            if (
                                collected[nworkers - 1][2] == collected[0][2]
                            ):  # ensure nworkers have provided the same num_steps
                                (
                                    last_train_steps,
                                    last_train_time,
                                ) = self.process_train_packages(
                                    log_writer,
                                    collected[:nworkers],
                                    last_steps=last_train_steps,
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
                    elif (
                        package[0] == "valid_package"
                    ):  # they all come from a single worker
                        if package[1] is not None:  # no validation samplers
                            self.process_eval_package(log_writer, package)
                        if (
                            finalized and self.queues["checkpoints"].empty()
                        ):  # assume queue is actually empty after trainer finished and no checkpoints in queue
                            break
                    elif package[0] == "test_package":
                        collected.append(package)
                        if len(collected) >= nworkers:
                            collected = sorted(
                                collected, key=lambda x: x[2]
                            )  # sort by num_steps
                            if (
                                collected[nworkers - 1][2] == collected[0][2]
                            ):  # ensure nworkers have provided the same num_steps
                                self.process_test_packages(
                                    log_writer, collected[:nworkers], test_results
                                )
                                collected = collected[nworkers:]
                                with open(metrics_file, "w") as f:
                                    json.dump(test_results, f, indent=4, sort_keys=True)
                                    get_logger().debug(
                                        "Updated {} up to checkpoint {}".format(
                                            metrics_file,
                                            test_steps[len(test_results) - 1],
                                        )
                                    )
                    elif package[0] == "train_stopped":
                        if package[1] == 0:
                            finalized = True
                        else:
                            raise Exception(
                                "Train worker {} abnormally terminated".format(
                                    package[1] - 1
                                )
                            )
                    elif package[0] == "valid_stopped":
                        raise Exception(
                            "Valid worker {} abnormally terminated".format(
                                package[1] - 1
                            )
                        )
                    elif package[0] == "test_stopped":
                        if package[1] == 0:
                            unfinished_workers -= 1
                            if unfinished_workers == 0:
                                get_logger().info("Last tester finished. Terminating")
                                finalized = True
                                break
                        else:
                            raise Exception(
                                "Test worker {} abnormally terminated".format(
                                    package[1] - 1
                                )
                            )
                    else:
                        get_logger().error(
                            "Runner received unknown package type {}".format(package[0])
                        )
                except queue.Empty as _:
                    if all(
                        p.exitcode is not None
                        for p in itertools.chain(*self.processes.values())
                    ):
                        break
        except KeyboardInterrupt:
            get_logger().info("KeyboardInterrupt. Terminating runner")
        except Exception:
            get_logger().error("Encountered Exception. Terminating runner")
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

    def step_from_checkpoint(self, name):
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
                    process.join(10)
                    logif("Closed {} {}".format(process_type, it))
                except Exception as e:
                    logif(
                        "Exception raised when closing {} {}".format(process_type, it)
                    )
                    logif(e)
                    pass

        self._is_closed = True

    def __del__(self):
        self.close(verbose=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(verbose=True)
