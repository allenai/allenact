# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import traceback
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from threading import Thread
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    Dict,
    Generator,
    Iterator,
    cast,
)

import numpy as np
from gym.spaces.dict import Dict as SpaceDict
from setproctitle import setproctitle as ptitle
import torch

from core.base_abstractions.misc import RLStepResult
from core.base_abstractions.task import TaskSampler
from utils.misc_utils import partition_sequence
from utils.system import get_logger
from utils.tensor_utils import tile_images

try:
    # Use torch.multiprocessing if we can.
    # We have yet to find a reason to not use it and
    # you are required to use it when sending a torch.Tensor
    # between processes
    import torch.multiprocessing as mp
except ImportError:
    import multiprocessing as mp  # type: ignore

DEFAULT_MP_CONTEXT_TYPE = "forkserver"
COMPLETE_TASK_METRICS_KEY = "__AFTER_TASK_METRICS__"

STEP_COMMAND = "step"
NEXT_TASK_COMMAND = "next_task"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
OBSERVATION_SPACE_COMMAND = "observation_space"
ACTION_SPACE_COMMAND = "action_space"
CALL_COMMAND = "call"
SAMPLER_COMMAND = "call_sampler"
ATTR_COMMAND = "attr"
SAMPLER_ATTR_COMMAND = "sampler_attr"
RESET_COMMAND = "reset"
SEED_COMMAND = "seed"
PAUSE_COMMAND = "pause"
RESUME_COMMAND = "resume"


class VectorSampledTasks(object):
    """Vectorized collection of tasks. Creates multiple processes where each
    process runs its own TaskSampler. Each process generates one Task from its
    TaskSampler at a time and this class allows for interacting with these
    tasks in a vectorized manner. When a task on a process completes, the
    process samples another task from its task sampler. All the tasks are
    synchronized (for step and new_task methods).

    # Attributes

    make_sampler_fn : function which creates a single TaskSampler.
    sampler_fn_args : sequence of dictionaries describing the args
        to pass to make_sampler_fn on each individual process.
    auto_resample_when_done : automatically sample a new Task from the TaskSampler when
        the Task completes. If False, a new Task will not be resampled until all
        Tasks on all processes have completed. This functionality is provided for seamless training
        of vectorized Tasks.
    multiprocessing_start_method : the multiprocessing method used to
        spawn worker processes. Valid methods are
        ``{'spawn', 'forkserver', 'fork'}`` ``'forkserver'`` is the
        recommended method as it works well with CUDA. If
        ``'fork'`` is used, the subproccess  must be started before
        any other GPU useage.
    """

    observation_space: SpaceDict
    _workers: List[Union[mp.Process, Thread, BaseProcess]]
    _is_waiting: bool
    _num_task_samplers: int
    _auto_resample_when_done: bool
    _mp_ctx: BaseContext
    _connection_read_fns: List[Callable[[], Any]]
    _connection_write_fns: List[Callable[[Any], None]]

    def __init__(
        self,
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args: Sequence[Dict[str, Any]] = None,
        auto_resample_when_done: bool = True,
        multiprocessing_start_method: Optional[str] = "forkserver",
        mp_ctx: Optional[BaseContext] = None,
        should_log: bool = True,
        max_processes: Optional[int] = None,
    ) -> None:

        self._is_waiting = False
        self._is_closed = True
        self.should_log = should_log
        self.max_processes = max_processes

        assert (
            sampler_fn_args is not None and len(sampler_fn_args) > 0
        ), "number of processes to be created should be greater than 0"

        self._num_task_samplers = len(sampler_fn_args)
        self._num_processes = (
            self._num_task_samplers
            if max_processes is None
            else min(max_processes, self._num_task_samplers)
        )

        self._auto_resample_when_done = auto_resample_when_done

        assert (multiprocessing_start_method is None) != (
            mp_ctx is None
        ), "Exactly one of `multiprocessing_start_method`, and `mp_ctx` must be not None."
        if multiprocessing_start_method is not None:
            assert multiprocessing_start_method in self._valid_start_methods, (
                "multiprocessing_start_method must be one of {}. Got '{}'"
            ).format(self._valid_start_methods, multiprocessing_start_method)
            self._mp_ctx = mp.get_context(multiprocessing_start_method)
        else:
            self._mp_ctx = cast(BaseContext, mp_ctx)

        self.npaused_per_process = [0] * self._num_processes
        self.sampler_index_to_process_ind_and_subprocess_ind: Optional[
            List[List[int]]
        ] = None
        self._reset_sampler_index_to_process_ind_and_subprocess_ind()

        self._workers: Optional[List] = None
        for args in sampler_fn_args:
            args["mp_ctx"] = self._mp_ctx
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            make_sampler_fn=make_sampler_fn,
            sampler_fn_args_list=[
                args_list for args_list in self._partition_to_processes(sampler_fn_args)
            ],
        )

        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((OBSERVATION_SPACE_COMMAND, None))

        observation_spaces = [
            space for read_fn in self._connection_read_fns for space in read_fn()
        ]

        if any(os is None for os in observation_spaces):
            raise NotImplementedError(
                "It appears that the `all_observation_spaces_equal`"
                " is not True for some task sampler created by"
                " VectorSampledTasks. This is not currently supported."
            )

        if any(observation_spaces[0] != os for os in observation_spaces):
            raise NotImplementedError(
                "It appears that the observation spaces of the samplers"
                " created in VectorSampledTasks are not equal."
                " This is not currently supported."
            )

        self.observation_space = observation_spaces[0]
        for write_fn in self._connection_write_fns:
            write_fn((ACTION_SPACE_COMMAND, None))
        self.action_spaces = [
            space for read_fn in self._connection_read_fns for space in read_fn()
        ]

    def _reset_sampler_index_to_process_ind_and_subprocess_ind(self):
        self.sampler_index_to_process_ind_and_subprocess_ind = [
            [i, j]
            for i, part in enumerate(
                partition_sequence([1] * self._num_task_samplers, self._num_processes)
            )
            for j in range(len(part))
        ]

    def _partition_to_processes(self, seq: Union[Iterator, Sequence]):
        subparts_list: List[List] = [[] for _ in range(self._num_processes)]

        seq = list(seq)
        assert len(seq) == len(self.sampler_index_to_process_ind_and_subprocess_ind)

        for sampler_index, (process_ind, subprocess_ind) in enumerate(
            self.sampler_index_to_process_ind_and_subprocess_ind
        ):
            assert len(subparts_list[process_ind]) == subprocess_ind
            subparts_list[process_ind].append(seq[sampler_index])

        return subparts_list

    @property
    def is_closed(self) -> bool:
        """Has the vector task been closed."""
        return self._is_closed

    @property
    def num_unpaused_tasks(self) -> int:
        """Number of unpaused processes.

        # Returns

        Number of unpaused processes.
        """
        return self._num_task_samplers - sum(self.npaused_per_process)

    @property
    def mp_ctx(self):
        """Get the multiprocessing process used by the vector task.

        # Returns

        The multiprocessing context.
        """
        return self._mp_ctx

    @staticmethod
    def _task_sampling_loop_worker(
        worker_id: Union[int, str],
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args_list: List[Dict[str, Any]],
        auto_resample_when_done: bool,
        should_log: bool,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        """process worker for creating and interacting with the
        Tasks/TaskSampler."""

        ptitle("VectorSampledTask: {}".format(worker_id))

        sp_vector_sampled_tasks = SingleProcessVectorSampledTasks(
            make_sampler_fn=make_sampler_fn,
            sampler_fn_args_list=sampler_fn_args_list,
            auto_resample_when_done=auto_resample_when_done,
            should_log=should_log,
        )

        if parent_pipe is not None:
            parent_pipe.close()
        try:
            while True:
                read_input = connection_read_fn()

                if len(read_input) == 3:
                    sampler_index, command, data = read_input

                    assert command != CLOSE_COMMAND, "Must close all processes at once."
                    assert (
                        command != RESUME_COMMAND
                    ), "Must resume all task samplers at once."

                    if command == PAUSE_COMMAND:
                        sp_vector_sampled_tasks.pause_at(sampler_index=sampler_index)
                        connection_write_fn("done")
                    else:
                        connection_write_fn(
                            sp_vector_sampled_tasks.command_at(
                                sampler_index=sampler_index, command=command, data=data
                            )
                        )
                else:
                    commands, data_list = read_input

                    assert (
                        commands != PAUSE_COMMAND
                    ), "Cannot pause all task samplers at once."

                    if commands == CLOSE_COMMAND:
                        sp_vector_sampled_tasks.close()
                        break
                    elif commands == RESUME_COMMAND:
                        sp_vector_sampled_tasks.resume_all()
                        connection_write_fn("done")
                    else:
                        if isinstance(commands, str):
                            commands = [
                                commands
                            ] * sp_vector_sampled_tasks.num_unpaused_tasks

                        connection_write_fn(
                            sp_vector_sampled_tasks.command(
                                commands=commands, data_list=data_list
                            )
                        )

            if child_pipe is not None:
                child_pipe.close()
        except KeyboardInterrupt as e:
            if should_log:
                get_logger().info("Worker {} KeyboardInterrupt".format(worker_id))
            raise e
        except Exception as e:
            get_logger().error(traceback.format_exc())
            raise e
        finally:
            if should_log:
                get_logger().info("""Worker {} closing.""".format(worker_id))

    def _spawn_workers(
        self,
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args_list: Sequence[Sequence[Dict[str, Any]]],
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_connections, worker_connections = zip(
            *[self._mp_ctx.Pipe(duplex=True) for _ in range(self._num_processes)]
        )
        self._workers = []
        k = 0
        id: Union[int, str]
        for id, stuff in enumerate(
            zip(worker_connections, parent_connections, sampler_fn_args_list)
        ):
            worker_conn, parent_conn, current_sampler_fn_args_list = stuff  # type: ignore

            if len(current_sampler_fn_args_list) != 1:
                id = "{}({}-{})".format(
                    id, k, k + len(current_sampler_fn_args_list) - 1
                )
                k += len(current_sampler_fn_args_list)

            if self.should_log:
                get_logger().info(
                    "Starting {}-th VectorSampledTask worker with args {}".format(
                        id, current_sampler_fn_args_list
                    )
                )
            ps = self._mp_ctx.Process(  # type: ignore
                target=self._task_sampling_loop_worker,
                args=(
                    id,
                    worker_conn.recv,
                    worker_conn.send,
                    make_sampler_fn,
                    current_sampler_fn_args_list,
                    self._auto_resample_when_done,
                    self.should_log,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(ps)
            ps.daemon = True
            ps.start()
            worker_conn.close()
            time.sleep(
                0.1
            )  # Useful to ensure things don't lock up when spawning many envs
        return (
            [p.recv for p in parent_connections],
            [p.send for p in parent_connections],
        )

    def next_task(self, **kwargs):
        """Move to the the next Task for all TaskSamplers.

        # Parameters

        kwargs : key word arguments passed to the `next_task` function of the samplers.

        # Returns

        List of initial observations for each of the new tasks.
        """
        return self.command(
            commands=NEXT_TASK_COMMAND, data_list=[kwargs] * self.num_unpaused_tasks
        )

    def get_observations(self):
        """Get observations for all unpaused tasks.

        # Returns

        List of observations for each of the unpaused tasks.
        """
        return self.call(["get_observations"] * self.num_unpaused_tasks,)

    def command_at(
        self, sampler_index: int, command: str, data: Optional[Any] = None
    ) -> Any:
        """Runs the command on the selected task and returns the result.

        # Parameters


        # Returns

        Result of the command.
        """
        self._is_waiting = True
        (
            process_ind,
            subprocess_ind,
        ) = self.sampler_index_to_process_ind_and_subprocess_ind[sampler_index]
        self._connection_write_fns[process_ind]((subprocess_ind, command, data))
        result = self._connection_read_fns[process_ind]()
        self._is_waiting = False
        return result

    def call_at(
        self,
        sampler_index: int,
        function_name: str,
        function_args: Optional[List[Any]] = None,
    ) -> Any:
        """Calls a function (which is passed by name) on the selected task and
        returns the result.

        # Parameters

        index : Which task to call the function on.
        function_name : The name of the function to call on the task.
        function_args : Optional function args.

        # Returns

        Result of calling the function.
        """
        return self.command_at(
            sampler_index=sampler_index,
            command=CALL_COMMAND,
            data=(function_name, function_args),
        )

    def next_task_at(self, sampler_index: int) -> List[RLStepResult]:
        """Move to the the next Task from the TaskSampler in index_process
        process in the vector.

        # Parameters

        index_process : Index of the process to be reset.

        # Returns

        List of length one containing the observations the newly sampled task.
        """
        return [
            self.command_at(
                sampler_index=sampler_index, command=NEXT_TASK_COMMAND, data=None
            )
        ]

    def step_at(self, sampler_index: int, action: torch.Tensor) -> List[RLStepResult]:
        """Step in the index_process task in the vector.

        # Parameters

        sampler_index : Index of the sampler to be reset.
        action : The action to take.

        # Returns

        List containing the output of step method on the task in the indexed process.
        """
        return [
            self.command_at(
                sampler_index=sampler_index, command=STEP_COMMAND, data=action
            )
        ]

    def async_step(self, actions: List[torch.Tensor]) -> None:
        """Asynchronously step in the vectorized Tasks.

        # Parameters

        actions : actions to be performed in the vectorized Tasks.
        """
        self._is_waiting = True
        for write_fn, action_tensor in zip(
            self._connection_write_fns, self._partition_to_processes(actions)
        ):
            write_fn((STEP_COMMAND, action_tensor))

    def wait_step(self) -> List[Dict[str, Any]]:
        """Wait until all the asynchronized processes have synchronized."""
        observations = []
        for read_fn in self._connection_read_fns:
            observations.extend(read_fn())
        self._is_waiting = False
        return observations

    def step(self, actions: List[torch.Tensor]):
        """Perform actions in the vectorized tasks.

        # Parameters

        actions: List of size _num_samplers containing action to be taken in each task.

        # Returns

        List of outputs from the step method of tasks.
        """
        self.async_step(actions)
        return self.wait_step()

    def reset_all(self):
        """Reset all task samplers to their initial state (except for the RNG
        seed)."""
        self.command(commands=RESET_COMMAND, data_list=None)

    def set_seeds(self, seeds: List[int]):
        """Sets new tasks' RNG seeds.

        # Parameters

        seeds: List of size _num_samplers containing new RNG seeds.
        """
        self.command(commands=SEED_COMMAND, data_list=seeds)

    def close(self) -> None:
        if self._is_closed:
            return

        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                try:
                    read_fn()
                except:
                    pass

        for write_fn in self._connection_write_fns:
            try:
                write_fn((CLOSE_COMMAND, None))
            except:
                pass

        for process in self._workers:
            try:
                process.join(timeout=0.1)
            except:
                pass

        self._is_closed = True

    def pause_at(self, sampler_index: int) -> None:
        """Pauses computation on the Task in process `index` without destroying
        the Task. This is useful for not needing to call steps on all Tasks
        when only some are active (for example during the last samples of
        running eval).

        # Parameters

        index : which process to pause. All indexes after this
            one will be shifted down by one.
        """
        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()

        (
            process_ind,
            subprocess_ind,
        ) = self.sampler_index_to_process_ind_and_subprocess_ind[sampler_index]

        self.command_at(sampler_index=sampler_index, command=PAUSE_COMMAND, data=None)

        for i in range(
            sampler_index + 1, len(self.sampler_index_to_process_ind_and_subprocess_ind)
        ):
            other_process_and_sub_process_inds = self.sampler_index_to_process_ind_and_subprocess_ind[
                i
            ]
            if other_process_and_sub_process_inds[0] == process_ind:
                other_process_and_sub_process_inds[1] -= 1
            else:
                break

        self.sampler_index_to_process_ind_and_subprocess_ind.pop(sampler_index)

        self.npaused_per_process[process_ind] += 1

    def resume_all(self) -> None:
        """Resumes any paused processes."""
        self._is_waiting = True
        for connection_write_fn in self._connection_write_fns:
            connection_write_fn((RESUME_COMMAND, None))

        for connection_read_fn in self._connection_read_fns:
            connection_read_fn()

        self._is_waiting = False

        self._reset_sampler_index_to_process_ind_and_subprocess_ind()

        for i in range(len(self.npaused_per_process)):
            self.npaused_per_process[i] = 0

    def command(
        self, commands: Union[List[str], str], data_list: Optional[List]
    ) -> List[Any]:
        """"""
        self._is_waiting = True

        if isinstance(commands, str):
            commands = [commands] * self.num_unpaused_tasks

        if data_list is None:
            data_list = [None] * self.num_unpaused_tasks

        for write_fn, subcommands, subdata_list in zip(
            self._connection_write_fns,
            self._partition_to_processes(commands),
            self._partition_to_processes(data_list),
        ):
            write_fn((subcommands, data_list))
        results = []
        for read_fn in self._connection_read_fns:
            results.extend(read_fn())
        self._is_waiting = False
        return results

    def call(
        self,
        function_names: Union[str, List[str]],
        function_args_list: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Calls a list of functions (which are passed by name) on the
        corresponding task (by index).

        # Parameters

        function_names : The name of the functions to call on the tasks.
        function_args_list : List of function args for each function.
            If provided, len(function_args_list) should be as long as  len(function_names).

        # Returns

        List of results of calling the functions.
        """
        self._is_waiting = True

        if isinstance(function_names, str):
            function_names = [function_names] * self.num_unpaused_tasks

        if function_args_list is None:
            function_args_list = [None] * len(function_names)
        assert len(function_names) == len(function_args_list)
        func_names_and_args_list = zip(function_names, function_args_list)
        for write_fn, func_names_and_args in zip(
            self._connection_write_fns,
            self._partition_to_processes(func_names_and_args_list),
        ):
            write_fn((CALL_COMMAND, func_names_and_args))
        results = []
        for read_fn in self._connection_read_fns:
            results.extend(read_fn())
        self._is_waiting = False
        return results

    def attr_at(self, sampler_index: int, attr_name: str) -> Any:
        """Gets the attribute (specified by name) on the selected task and
        returns it.

        # Parameters

        index : Which task to call the function on.
        attr_name : The name of the function to call on the task.

        # Returns

         Result of calling the function.
        """
        return self.command_at(sampler_index, command=ATTR_COMMAND, data=attr_name)

    def attr(self, attr_names: Union[List[str], str]) -> List[Any]:
        """Gets the attributes (specified by name) on the tasks.

        # Parameters

        attr_names : The name of the functions to call on the tasks.

        # Returns

        List of results of calling the functions.
        """
        if isinstance(attr_names, str):
            attr_names = [attr_names] * self.num_unpaused_tasks

        return self.command(commands=ATTR_COMMAND, data_list=attr_names)

    def render(
        self, mode: str = "human", *args, **kwargs
    ) -> Union[np.ndarray, None, List[np.ndarray]]:
        """Render observations from all Tasks in a tiled image or list of
        images."""

        images = self.command(
            commands=RENDER_COMMAND,
            data_list=[(args, {"mode": "rgb", **kwargs})] * self.num_unpaused_tasks,
        )

        if mode == "raw_rgb_list":
            return images

        tile = tile_images(images)
        if mode == "human":
            import cv2

            cv2.imshow("vectask", tile[:, :, ::-1])
            cv2.waitKey(1)
            return None
        elif mode == "rgb_array":
            return tile
        else:
            raise NotImplementedError

    @property
    def _valid_start_methods(self) -> Set[str]:
        return {"forkserver", "spawn", "fork"}

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SingleProcessVectorSampledTasks(object):
    """Vectorized collection of tasks.

    Simultaneously handles the state of multiple TaskSamplers and their associated tasks.
    Allows for interacting with these tasks in a vectorized manner. When a task completes,
    another task is sampled from the appropriate task sampler. All the tasks are
    synchronized (for step and new_task methods).

    # Attributes

    make_sampler_fn : function which creates a single TaskSampler.
    sampler_fn_args : sequence of dictionaries describing the args
        to pass to make_sampler_fn on each individual process.
    auto_resample_when_done : automatically sample a new Task from the TaskSampler when
        the Task completes. If False, a new Task will not be resampled until all
        Tasks on all processes have completed. This functionality is provided for seamless training
        of vectorized Tasks.
    """

    observation_space: SpaceDict
    _vector_task_generators: List[Generator]
    _num_task_samplers: int
    _auto_resample_when_done: bool

    def __init__(
        self,
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args_list: Sequence[Dict[str, Any]] = None,
        auto_resample_when_done: bool = True,
        should_log: bool = True,
    ) -> None:

        self._is_closed = True

        assert (
            sampler_fn_args_list is not None and len(sampler_fn_args_list) > 0
        ), "number of processes to be created should be greater than 0"

        self._num_task_samplers = len(sampler_fn_args_list)
        self._auto_resample_when_done = auto_resample_when_done

        self.should_log = should_log

        self._vector_task_generators: List[Generator] = self._create_generators(
            make_sampler_fn=make_sampler_fn,
            sampler_fn_args=[{"mp_ctx": None, **args} for args in sampler_fn_args_list],
        )

        self._is_closed = False

        observation_spaces = [
            vsi.send((OBSERVATION_SPACE_COMMAND, None))
            for vsi in self._vector_task_generators
        ]

        if any(os is None for os in observation_spaces):
            raise NotImplementedError(
                "It appears that the `all_observation_spaces_equal`"
                " is not True for some task sampler created by"
                " VectorSampledTasks. This is not currently supported."
            )

        if any(observation_spaces[0] != os for os in observation_spaces):
            raise NotImplementedError(
                "It appears that the observation spaces of the samplers"
                " created in VectorSampledTasks are not equal."
                " This is not currently supported."
            )

        self.observation_space = observation_spaces[0]
        self.action_spaces = [
            vsi.send((ACTION_SPACE_COMMAND, None))
            for vsi in self._vector_task_generators
        ]
        self._paused: List[Tuple[int, Generator]] = []

    @property
    def is_closed(self) -> bool:
        """Has the vector task been closed."""
        return self._is_closed

    @property
    def mp_ctx(self) -> Optional[BaseContext]:
        return None

    @property
    def num_unpaused_tasks(self) -> int:
        """Number of unpaused processes.

        # Returns

        Number of unpaused processes.
        """
        return self._num_task_samplers - len(self._paused)

    @staticmethod
    def _task_sampling_loop_generator_fn(
        worker_id: int,
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args: Dict[str, Any],
        auto_resample_when_done: bool,
        should_log: bool,
    ) -> Generator:
        """Generator for working with Tasks/TaskSampler."""

        task_sampler = make_sampler_fn(**sampler_fn_args)
        current_task = task_sampler.next_task()

        if current_task is None:
            raise RuntimeError(
                "Newly created task sampler had `None` as it's first task. This likely means that"
                " it was not provided with any tasks to generate. This can happen if, e.g., during testing"
                " you have started more processes than you had tasks to test. Currently this is not supported:"
                " every task sampler must be able to generate at least one task."
            )

        try:
            command, data = yield "started"

            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    # # TODO Adding this for backward compatibility with existing tasks. Would be best to just send data.
                    # if len(data) == 1:
                    #     data = data[0]
                    step_result: RLStepResult = current_task.step(data)
                    if current_task.is_done():
                        metrics = current_task.metrics()
                        if metrics is not None and len(metrics) != 0:
                            step_result.info[COMPLETE_TASK_METRICS_KEY] = metrics

                        if auto_resample_when_done:
                            current_task = task_sampler.next_task()
                            if current_task is None:
                                step_result = step_result.clone({"observation": None})
                            else:
                                step_result = step_result.clone(
                                    {"observation": current_task.get_observations()}
                                )

                    command, data = yield step_result

                elif command == NEXT_TASK_COMMAND:
                    if data is not None:
                        current_task = task_sampler.next_task(**data)
                    else:
                        current_task = task_sampler.next_task()
                    observations = current_task.get_observations()

                    command, data = yield observations

                elif command == RENDER_COMMAND:
                    command, data = yield current_task.render(*data[0], **data[1])

                elif (
                    command == OBSERVATION_SPACE_COMMAND
                    or command == ACTION_SPACE_COMMAND
                ):
                    res = getattr(current_task, command)
                    command, data = yield res

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None or len(function_args) == 0:
                        result = getattr(current_task, function_name)()
                    else:
                        result = getattr(current_task, function_name)(*function_args)
                    command, data = yield result

                elif command == SAMPLER_COMMAND:
                    function_name, function_args = data
                    if function_args is None or len(function_args) == 0:
                        result = getattr(task_sampler, function_name)()
                    else:
                        result = getattr(task_sampler, function_name)(*function_args)

                    command, data = yield result

                elif command == ATTR_COMMAND:
                    property_name = data
                    result = getattr(current_task, property_name)

                    command, data = yield result

                elif command == SAMPLER_ATTR_COMMAND:
                    property_name = data
                    result = getattr(task_sampler, property_name)

                    command, data = yield result

                elif command == RESET_COMMAND:
                    task_sampler.reset()
                    current_task = task_sampler.next_task()

                    if current_task is None:
                        raise RuntimeError(
                            "After resetting the task sampler it seems to have"
                            " no new tasks (the `task_sampler.next_task()` call"
                            " returned `None` after the reset). This suggests that"
                            " the task sampler's reset method was not implemented"
                            f" correctly (task sampler type is {type(task_sampler)})."
                        )

                    command, data = yield "done"
                elif command == SEED_COMMAND:
                    task_sampler.set_seed(data)

                    command, data = yield "done"
                else:
                    raise NotImplementedError()

        except KeyboardInterrupt:
            if should_log:
                get_logger().info(
                    "SingleProcessVectorSampledTask {} KeyboardInterrupt".format(
                        worker_id
                    )
                )
        except Exception as e:
            get_logger().error(traceback.format_exc())
            raise e
        finally:
            if should_log:
                get_logger().info(
                    "SingleProcessVectorSampledTask {} closing.".format(worker_id)
                )
            task_sampler.close()

    def _create_generators(
        self,
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args: Sequence[Dict[str, Any]],
    ) -> List[Generator]:

        generators = []
        for id, current_sampler_fn_args in enumerate(sampler_fn_args):
            if self.should_log:
                get_logger().info(
                    "Starting {}-th SingleProcessVectorSampledTasks generator with args {}".format(
                        id, current_sampler_fn_args
                    )
                )
            generators.append(
                self._task_sampling_loop_generator_fn(
                    worker_id=id,
                    make_sampler_fn=make_sampler_fn,
                    sampler_fn_args=current_sampler_fn_args,
                    auto_resample_when_done=self._auto_resample_when_done,
                    should_log=self.should_log,
                )
            )

            if next(generators[-1]) != "started":
                raise RuntimeError("Generator failed to start.")

        return generators

    def next_task(self, **kwargs):
        """Move to the the next Task for all TaskSamplers.

        # Parameters

        kwargs : key word arguments passed to the `next_task` function of the samplers.

        # Returns

        List of initial observations for each of the new tasks.
        """
        return [
            g.send((NEXT_TASK_COMMAND, kwargs)) for g in self._vector_task_generators
        ]

    def get_observations(self):
        """Get observations for all unpaused tasks.

        # Returns

        List of observations for each of the unpaused tasks.
        """
        return self.call(["get_observations"] * self.num_unpaused_tasks,)

    def next_task_at(self, index_process: int) -> List[RLStepResult]:
        """Move to the the next Task from the TaskSampler in index_process
        process in the vector.

        # Parameters

        index_process : Index of the generator to be reset.

        # Returns

        List of length one containing the observations the newly sampled task.
        """
        return [
            self._vector_task_generators[index_process].send((NEXT_TASK_COMMAND, None))
        ]

    def step_at(self, index_process: int, action: int) -> List[RLStepResult]:
        """Step in the index_process task in the vector.

        # Parameters

        index_process : Index of the process to be reset.
        action : The action to take.

        # Returns

        List containing the output of step method on the task in the indexed process.
        """
        return self._vector_task_generators[index_process].send((STEP_COMMAND, action))

    def step(self, actions: List[List[int]]):
        """Perform actions in the vectorized tasks.

        # Parameters

        actions: List of size _num_samplers containing action to be taken in each task.

        # Returns

        List of outputs from the step method of tasks.
        """
        return [
            g.send((STEP_COMMAND, action))
            for g, action in zip(self._vector_task_generators, actions)
        ]

    def reset_all(self):
        """Reset all task samplers to their initial state (except for the RNG
        seed)."""
        return [g.send((RESET_COMMAND, None)) for g in self._vector_task_generators]

    def set_seeds(self, seeds: List[int]):
        """Sets new tasks' RNG seeds.

        # Parameters

        seeds: List of size _num_samplers containing new RNG seeds.
        """
        return [
            g.send((SEED_COMMAND, seed))
            for g, seed in zip(self._vector_task_generators, seeds)
        ]

    def close(self) -> None:
        if self._is_closed:
            return

        for g in self._vector_task_generators:
            try:
                g.send((CLOSE_COMMAND, None))
            except StopIteration:
                pass

        self._is_closed = True

    def pause_at(self, sampler_index: int) -> None:
        """Pauses computation on the Task in process `index` without destroying
        the Task. This is useful for not needing to call steps on all Tasks
        when only some are active (for example during the last samples of
        running eval).

        # Parameters

        index : which process to pause. All indexes after this
            one will be shifted down by one.
        """
        generator = self._vector_task_generators.pop(sampler_index)
        self._paused.append((sampler_index, generator))

    def resume_all(self) -> None:
        """Resumes any paused processes."""
        for index, generator in reversed(self._paused):
            self._vector_task_generators.insert(index, generator)
        self._paused = []

    def command_at(
        self, sampler_index: int, command: str, data: Optional[Any] = None
    ) -> Any:
        """Calls a function (which is passed by name) on the selected task and
        returns the result.

        # Parameters

        index : Which task to call the function on.
        function_name : The name of the function to call on the task.
        function_args : Optional function args.

        # Returns

        Result of calling the function.
        """
        return self._vector_task_generators[sampler_index].send((command, data))

    def command(
        self, commands: Union[List[str], str], data_list: Optional[List]
    ) -> List[Any]:
        """"""
        if isinstance(commands, str):
            commands = [commands] * self.num_unpaused_tasks

        if data_list is None:
            data_list = [None] * self.num_unpaused_tasks

        return [
            g.send((command, data))
            for g, command, data in zip(
                self._vector_task_generators, commands, data_list
            )
        ]

    def call_at(
        self,
        sampler_index: int,
        function_name: str,
        function_args: Optional[List[Any]] = None,
    ) -> Any:
        """Calls a function (which is passed by name) on the selected task and
        returns the result.

        # Parameters

        index : Which task to call the function on.
        function_name : The name of the function to call on the task.
        function_args : Optional function args.

        # Returns

        Result of calling the function.
        """
        return self._vector_task_generators[sampler_index].send(
            (CALL_COMMAND, (function_name, function_args))
        )

    def call(
        self,
        function_names: Union[str, List[str]],
        function_args_list: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Calls a list of functions (which are passed by name) on the
        corresponding task (by index).

        # Parameters

        function_names : The name of the functions to call on the tasks.
        function_args_list : List of function args for each function.
            If provided, len(function_args_list) should be as long as  len(function_names).

        # Returns

        List of results of calling the functions.
        """
        if isinstance(function_names, str):
            function_names = [function_names] * self.num_unpaused_tasks

        if function_args_list is None:
            function_args_list = [None] * len(function_names)

        assert len(function_names) == len(function_args_list)

        return [
            g.send((CALL_COMMAND, args))
            for g, args in zip(
                self._vector_task_generators, zip(function_names, function_args_list)
            )
        ]

    def attr_at(self, sampler_index: int, attr_name: str) -> Any:
        """Gets the attribute (specified by name) on the selected task and
        returns it.

        # Parameters

        index : Which task to call the function on.
        attr_name : The name of the function to call on the task.

        # Returns

         Result of calling the function.
        """
        return self._vector_task_generators[sampler_index].send(
            (ATTR_COMMAND, attr_name)
        )

    def attr(self, attr_names: Union[List[str], str]) -> List[Any]:
        """Gets the attributes (specified by name) on the tasks.

        # Parameters

        attr_names : The name of the functions to call on the tasks.

        # Returns

        List of results of calling the functions.
        """
        if isinstance(attr_names, str):
            attr_names = [attr_names] * self.num_unpaused_tasks

        return [
            g.send((ATTR_COMMAND, attr_name))
            for g, attr_name in zip(self._vector_task_generators, attr_names)
        ]

    def render(
        self, mode: str = "human", *args, **kwargs
    ) -> Union[np.ndarray, None, List[np.ndarray]]:
        """Render observations from all Tasks in a tiled image or a list of
        images."""

        images = [
            g.send((RENDER_COMMAND, (args, {"mode": "rgb", **kwargs})))
            for g in self._vector_task_generators
        ]

        if mode == "raw_rgb_list":
            return images

        for index, _ in reversed(self._paused):
            images.insert(index, np.zeros_like(images[0]))

        tile = tile_images(images)
        if mode == "human":
            import cv2

            cv2.imshow("vectask", tile[:, :, ::-1])
            cv2.waitKey(1)
            return None
        elif mode == "rgb_array":
            return tile
        else:
            raise NotImplementedError

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
