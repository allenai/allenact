"""Defines the reinforcement learning `OnPolicyRLEngine`."""
import glob
import os
import queue
import random
import shutil
import sys
import time
import traceback
import typing
import warnings
from multiprocessing.context import BaseContext
from typing import Optional, Any, Dict, Union, List, Tuple, Set, Callable
import json
from collections import OrderedDict, defaultdict
import signal

import torch
import torch.distributions
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
from setproctitle import setproctitle as ptitle
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

from onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from onpolicy_sync.storage import RolloutStorage
from onpolicy_sync.vector_sampled_tasks import VectorSampledTasks, ThreadedVectorSampledTasks
from onpolicy_sync.vector_preprocessed_tasks import VectorPreprocessedTasks, ThreadedVectorPreprocessedTasks
from rl_base.experiment_config import ExperimentConfig
from utils.experiment_utils import (
    ScalarMeanTracker,
    LinearDecay,
    set_deterministic_cudnn,
    set_seed,
    Builder,
    TrainingPipeline,
    PipelineStage,
)
from utils.tensor_utils import batch_observations, SummaryWriter, tensor_to_video
from utils.system import LOGGER, init_logging, find_free_port
from onpolicy_sync.light_engine import OnPolicyRLEngine, OnPolicyTrainer, OnPolicyInference


# Has results queue (aggregated per trainer), checkpoints queue and mp context
# Instantiates train, validate, and test workers
# Logging
# Saves configs, makes folder for trainer models
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
        self.mode = self.init_mode(mode)
        self.extra_tag = extra_tag

        if self.deterministic_cudnn:
            set_deterministic_cudnn()

        if self.seed is not None:
            set_seed(self.seed)

        self.queues = {
            "results": self.mp_ctx.Queue(),
            "checkpoints": self.mp_ctx.Queue()
        }

        self.processes: Dict[str, list[mp.Process]] = defaultdict(list)

        self.current_checkpoint = None

        self.local_start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

        self.scalars = ScalarMeanTracker()

        self._is_closed: bool = False

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
            LOGGER.warning("ignoring multiprocessing_start_method '{}' and using given context with '{}'".format(
                multiprocessing_start_method,
                mp_ctx.get_start_method()
            ))

        return mp_ctx

    @staticmethod
    def init_mode(mode: str="train", supported: Tuple[str, ...] = ("train", "test")):
        mode = mode.lower()
        assert mode in supported, "Only {} modes supported in runner".format(supported)
        return mode

    @staticmethod
    def worker_devices(machine_params):
        devices = [torch.device("cpu")]
        LOGGER.debug("requested GPUs {}".format(machine_params["gpu_ids"]))
        if len(machine_params["gpu_ids"]) > 0:
            assert all([id >= 0 for id in machine_params["gpu_ids"]]), "all gpu_ids must be >= 0"
            assert torch.cuda.device_count() > max(set(machine_params["gpu_ids"])),\
                "{} CUDA devices available for requested gpu ids {}".format(
                        torch.cuda.device_count(), machine_params["gpu_ids"]
                    )
            # devices = [torch.device("cuda:{}".format(id)) for id in machine_params["gpu_ids"]]
            devices = [id for id in machine_params["gpu_ids"]]
        LOGGER.info("Using {} workers on devices {}".format(len(devices), devices))
        return devices

    @staticmethod
    def init_process(mode, id):
        ptitle("{}-{}".format(mode, id))

        def sigterm_handler(_signo, _stack_frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, sigterm_handler)

        init_logging()

    @staticmethod
    def start_train_loop(id: int=0, checkpoint: Optional[str]=None, *engine_args, **engine_kwargs):
        OnPolicyRunner.init_process("Train", id)
        engine_kwargs["mode"] = "train"
        engine_kwargs["worker_id"] = id
        LOGGER.info("train {} args {}".format(id, engine_kwargs))
        trainer = OnPolicyTrainer(*engine_args, **engine_kwargs)
        trainer.run_pipeline(checkpoint)

    @staticmethod
    def start_valid_loop(id: int=0, *engine_args, **engine_kwargs):
        OnPolicyRunner.init_process("Valid", id)
        engine_kwargs["mode"] = "valid"
        engine_kwargs["worker_id"] = id
        LOGGER.info("valid {} args {}".format(id, engine_kwargs))
        valid = OnPolicyInference(*engine_args, **engine_kwargs)
        valid.process_checkpoints()  # gets checkpoints via queue

    @staticmethod
    def start_test_loop(id: int=0, *engine_args, **engine_kwargs):
        OnPolicyRunner.init_process("Test", id)
        engine_kwargs["mode"] = "test"
        engine_kwargs["worker_id"] = id
        LOGGER.info("test {} args {}".format(id, engine_kwargs))
        test = OnPolicyInference(*engine_args, **engine_kwargs)
        test.process_checkpoints()  # gets checkpoints via queue

    def start_train(self, checkpoint: Optional[str] = None):
        self.save_config_files()

        machine_params = self.config.machine_params("train")  # TODO make sure nothing (preprocessors, etc) is instantiated in this call!!!
        devices = self.worker_devices(machine_params)
        nworkers = len(devices)

        seed = self.seed  # same for all workers. used during initialization of the model

        distributed_port = 0
        distributed_barrier = None
        if nworkers > 1:
            distributed_port = find_free_port()
            distributed_barrier = self.mp_ctx.Barrier(nworkers)

        for wit in range(nworkers):
            train: mp.Process = self.mp_ctx.Process(
                target=self.start_train_loop,
                args=(wit, checkpoint),
                kwargs=dict(
                    experiment_name=self.experiment_name,
                    config=self.config,
                    results_queue=self.queues["results"],
                    checkpoints_queue=self.queues["checkpoints"],
                    checkpoints_dir=self.checkpoint_dir,
                    seed=seed,
                    deterministic_cudnn=self.deterministic_cudnn,
                    mp_ctx=self.mp_ctx,
                    num_workers=nworkers,
                    device=devices[wit],
                    distributed_port=distributed_port,
                    distributed_barrier=distributed_barrier
                )
            )
            train.start()
            self.processes['train'].append(train)

        LOGGER.info('Started {} train processes'.format(len(self.processes['train'])))

        # Validation
        # TODO assume always single process?
        machine_params = self.config.machine_params("valid")
        device = self.worker_devices(machine_params)[0]
        valid: mp.Process = self.mp_ctx.Process(
            target=self.start_valid_loop,
            args=(0,),
            kwargs=dict(
                config=self.config,
                results_queue=self.queues["results"],
                checkpoints_queue=self.queues["checkpoints"],
                seed=12345,  # TODO allow same order for randomly sampled tasks? Is this any useful anyway?
                mp_ctx=self.mp_ctx,
                device=device,
            )
        )
        valid.start()
        self.processes['valid'].append(valid)

        LOGGER.info('Started {} valid processes'.format(len(self.processes['valid'])))

        self.log(nworkers)

    @staticmethod
    def checkpoint_start_time_str(checkpoint_file_name):
        parts = checkpoint_file_name.split(os.path.sep)
        assert len(parts) > 1, "{} is not a valid path)".format(checkpoint_file_name)
        start_time_str = parts[-2]
        LOGGER.info("Using start time {}".format(start_time_str))
        return start_time_str

    @property
    def experiment_name(self):  # only used by train
        if len(self.extra_tag) > 0:
            return "{}_{}".format(self.config.tag(), self.extra_tag)
        return self.config.tag()


    @property
    def checkpoint_dir(self):  # only used by train
        folder = os.path.join(
            self.output_dir,
            "checkpoints",
            self.local_start_time_str
        )
        os.makedirs(folder, exist_ok=True)
        return folder

    def log_writer_path(self, local_start_time_str) -> str:
        return os.path.join(
            self.output_dir,
            "tb",
            self.experiment_name,
            local_start_time_str,
        )

    def save_config_files(self):
        basefolder = os.path.join(
                    self.output_dir,
                    "used_configs",
                    self.local_start_time_str
        )

        for file in self.loaded_config_src_files:
            base, module = self.loaded_config_src_files[file]
            parts = module.split(".")

            src_file = os.path.sep.join([base] + parts) + ".py"
            if not os.path.isfile(src_file):
                LOGGER.error("Config file {} not found".format(src_file))

            dst_file = (
                os.path.join(
                    basefolder,
                    os.path.join(*parts[1:]),
                )
                + ".py"
            )
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)

            shutil.copy(src_file, dst_file)

        LOGGER.info("Config files saved to {}".format(basefolder))

    def process_eval_package(self, log_writer, pkg):
        pkg_type, payload, steps = pkg
        metrics_pkg, task_outputs, render, checkpoint_file_name = payload

        metrics_type, metrics_payload, num_tasks = metrics_pkg

        mode = pkg_type.split("_")[0]

        metrics = OrderedDict(
            sorted(
                [(k, v) for k, v in metrics_payload.items() if k != "task_info"], key=lambda x: x[0]
            )
        )

        message = ["{} {} steps:".format(mode, steps)]
        for k in metrics:
            log_writer.add_scalar("{}/".format(mode) + k, metrics[k], steps)
            message.append(k + " {}".format(metrics[k]))
        message.append("tasks {} checkpoint {}".format(num_tasks, checkpoint_file_name))
        LOGGER.info(" ".join(message))

        if render is not None:
            log_writer.add_vid("{}/agent_view".format(mode), render, steps)

    def aggregate_infos(self, log_writer, infos, steps):
        nsamples = sum(info[2] for info in infos)

        # assert nsamples != 0, "Attempting to aggregate infos with 0 samples".format(type)
        assert self.scalars.empty, "Attempting to aggregate with non-empty ScalarMeanTracker"

        for name, payload, nsamps in infos:
            assert nsamps >= 0, "negative ({}) samples in info".format(nsamps)
            if nsamps > 0:
                self.scalars.add_scalars(
                    {k: len(infos) * payload[k] * nsamps / nsamples for k in payload}
                )

        message = []
        if nsamples > 0:
            summary = self.scalars.pop_and_reset()

            metrics = OrderedDict(
                sorted(
                    [(k, v) for k, v in summary.items() if k != "task_info"], key=lambda x: x[0]
                )
            )

            for k in metrics:
                log_writer.add_scalar("{}/".format(self.mode) + k, metrics[k], steps)
                message.append(k + " {}".format(metrics[k]))

        return message

    def process_train_packages(self, log_writer, pkgs):
        pkg_types, payloads, all_steps = [vals for vals in zip(*pkgs)]

        steps = all_steps[0]

        all_info_types = [worker_pkgs for worker_pkgs in zip(*payloads)]

        message = ["train {} steps:".format(steps)]
        for info_type in all_info_types:
            message += self.aggregate_infos(log_writer, info_type, steps)
        LOGGER.info(" ".join(message))

    def log(self, nworkers, checkpoints=None):
        if checkpoints is None:
            checkpoints = []
        finalized = False

        if len(checkpoints) == 0:
            folder = self.local_start_time_str
        else:
            folder = self.checkpoint_start_time_str(checkpoints[0])

        log_writer = SummaryWriter(
            log_dir=self.log_writer_path(folder),
            filename_suffix="__{}_{}".format(self.mode, self.local_start_time_str),
        )

        try:
            collected = []
            while True:
                try:
                    package = self.queues["results"].get(timeout=10)  # TODO wait for 10 seconds, then check all workers are alive
                except queue.Empty:
                    for process_type in self.processes:
                        for it, process in enumerate(self.processes[process_type]):
                            assert process.is_alive(), "Process {} {} dead!".format(process_type, it)
                    continue
                if package[0] == "train_package":
                    collected.append(package)
                    if len(collected) >= nworkers:
                        collected = sorted(collected, key=lambda x: x[2])  # sort by num_steps
                        if collected[nworkers - 1][2] == collected[0][2]:  # ensure nworkers have provided the same num_steps
                            self.process_train_packages(log_writer, collected[:nworkers])
                            collected = collected[nworkers:]
                        elif len(collected) > 2 * nworkers:
                            raise Exception("Unable to aggregate train packages from {} workers".format(nworkers))
                elif package[0] == "valid_package":
                    if package[1] is None:  # no validation samplers
                        pass
                    self.process_eval_package(log_writer, package)
                    if finalized and self.queues["checkpoints"].empty():  # assume queue is actually empty after trainer finished and no checkpoints in queue
                        break
                elif package[0] == "test_package":
                    pass
                elif package[0] == "training_stopped":
                    if package[1] == 0:
                        finalized = True
                    else:
                        raise Exception("Train worker {} abnormally terminated".format(package[1] - 1))
                else:
                    LOGGER.warning("Runner received unknown package type {}".format(package[0]))
        except KeyboardInterrupt:
            LOGGER.info("KeyboardInterrupt. Terminating runner")
        except Exception:
            LOGGER.error("Encountered Exception. Terminating runner")
            LOGGER.exception(traceback.format_exc())
        finally:
            if finalized:
                LOGGER.info("Done")
            if log_writer is not None:
                log_writer.close()
            self.close()

    def start_test(self,
                   test_date: Optional[str] = None,
                   checkpoint: Optional[str] = None,
                   skip_checkpoints: int = 0,
                   ):
        pass

    # # TODO
    # def start_test(self,
    #                test_date: Optional[str] = None,
    #                checkpoint: Optional[str] = None,
    #                skip_checkpoints: int = 0,
    #                ):
    #     assert self.mode == "test", "run_test only to be called from a test instance"
    #
    #     test_start_time_str = time.strftime(
    #         "%Y-%m-%d_%H-%M-%S", time.localtime(time.time())
    #     )
    #
    #     self.local_start_time_str = experiment_date
    #
    #     checkpoints = self.get_checkpoint_files(
    #         experiment_date, checkpoint_file_name, skip_checkpoints
    #     )
    #
    #     suffix = "__test_{}".format(test_start_time_str)
    #     self.log_writer = SummaryWriter(
    #         log_dir=self.log_writer_path, filename_suffix=suffix,
    #     )
    #
    #     os.makedirs(self.metric_path, exist_ok=True)
    #     fname = os.path.join(self.metric_path, "metrics" + suffix + ".json")
    #
    #     LOGGER.info("Saving metrics in {}".format(fname))
    #
    #     all_results = []
    #     for it, checkpoint_file_name in enumerate(checkpoints):
    #         step = self.step_from_checkpoint(checkpoint_file_name)
    #         LOGGER.info("{}/{} {} steps".format(it + 1, len(checkpoints), step,))
    #
    #         scalars, render, samples = self.run_eval(
    #             checkpoint_file_name, rollout_steps
    #         )
    #
    #         self.vector_tasks.metrics_out_queue.put(("test_metrics", (scalars, render)))
    #
    #         results = {scalar: scalars[scalar][0] for scalar in scalars}
    #         results.update({"training_steps": step, "tasks": samples})
    #         all_results.append(results)
    #
    #         with open(fname, "w") as f:
    #             json.dump(all_results, f, indent=4)
    #
    #         self.log(count=1)
    #
    #         with open(fname, "w") as f:
    #             json.dump(all_results, f, indent=4)
    #
    #     LOGGER.info("Metrics saved in {}".format(fname))


    # @property
    # def metric_path(self) -> str:
    #     return os.path.join(
    #         self.output_dir, "metrics", self.experiment_name, self.local_start_time_str
    #     )

    # def get_checkpoint_path(self, checkpoint_file_name: str) -> str:
    #     checkpoint_start_time = [
    #         s for s in checkpoint_file_name.split("__") if "time_" in s
    #     ][0].replace("time_", "")
    #
    #     expected_path = os.path.join(
    #         self.output_dir, "checkpoints", checkpoint_start_time, checkpoint_file_name
    #     )
    #     if os.path.exists(expected_path):
    #         return expected_path
    #     else:
    #         print(
    #             (
    #                 "Could not find checkpoint with file name {}\n"
    #                 "under expected path {}.\n"
    #                 "Attempting to find the checkpoint elsewhere under the working directory.\n"
    #             ).format(checkpoint_file_name, expected_path)
    #         )
    #
    #         ckpts = glob.glob("./**/{}".format(checkpoint_file_name), recursive=True)
    #
    #         if len(ckpts) == 0:
    #             raise RuntimeError(
    #                 "Could not find {} anywhere"
    #                 " the working directory.".format(checkpoint_file_name)
    #             )
    #         elif len(ckpts) > 1:
    #             raise RuntimeError("Found too many checkpoint paths {}.".format(ckpts))
    #         else:
    #             return ckpts[0]

    # def get_checkpoint_files(
    #     self,
    #     experiment_date: str,
    #     checkpoint_file_name: Optional[str] = None,
    #     skip_checkpoints: int = 0,
    # ):
    #     if checkpoint_file_name is not None:
    #         return [checkpoint_file_name]
    #     files = glob.glob(
    #         os.path.join(self.output_dir, "checkpoints", experiment_date, "exp_*.pt")
    #     )
    #     files = sorted(files)
    #     return (
    #         files[:: skip_checkpoints + 1]
    #         + (
    #             [files[-1]]
    #             if skip_checkpoints > 0 and len(files) % (skip_checkpoints + 1) != 1
    #             else []
    #         )
    #         if len(files) > 0
    #         else files
    #     )

    # def step_from_checkpoint(self, name):
    #     parts = name.split("__")
    #     for part in parts:
    #         if "steps_" in part:
    #             return int(part.split("_")[-1])
    #     return -1

    def close(self, verbose=True):
        if self._is_closed:
            return

        def logif(s: Union[str, Exception]):
            if verbose:
                if isinstance(s, str):
                    LOGGER.info(s)
                elif isinstance(s, Exception):
                    LOGGER.exception(traceback.format_exc())
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
                    logif("Exception raised when closing {} {}".format(process_type, it))
                    logif(e)
                    pass

        self._is_closed = True

    def __del__(self):
        self.close(verbose=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(verbose=True)
