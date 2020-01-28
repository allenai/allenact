#!/usr/bin/env python3

# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Thread
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union, Dict

import numpy as np
from gym.spaces.dict import Dict as SpaceDict

from rl_base.common import RLStepResult
from rl_base.task import TaskSampler

try:
    # Use torch.multiprocessing if we can.
    # We have yet to find a reason to not use it and
    # you are required to use it when sending a torch.Tensor
    # between processes
    import torch.multiprocessing as mp
except ImportError:
    import multiprocessing as mp

STEP_COMMAND = "step"
NEXT_TASK_COMMAND = "next_task"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
OBSERVATION_SPACE_COMMAND = "observation_space"
ACTION_SPACE_COMMAND = "action_space"
CALL_COMMAND = "call"
ATTR_COMMAND = "attr"
# EPISODE_COMMAND = "current_episode"
RESET_COMMAND = "reset"


def tile_images(images: List[np.ndarray]) -> np.ndarray:
    """Tile multiple images into single image.

    @param images: list of images where each image has dimension
        (height x width x channels)
    @return: tiled image (new_height x width x channels)
    """
    assert len(images) > 0, "empty list of images"
    np_images = np.asarray(images)
    n_images, height, width, n_channels = np_images.shape
    new_height = int(np.ceil(np.sqrt(n_images)))
    new_width = int(np.ceil(float(n_images) / new_height))
    # pad with empty images to complete the rectangle
    np_images = np.array(
        images + [images[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = np_images.reshape((new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image


class VectorSampledTasks:
    """Vectorized collection of tasks. Creates multiple processes where each
    process runs its own TaskSampler. Each process generates one Task from its
    TaskSampler at a time and this class allows for interacting with these
    tasks in a vectorized manner. When a task on a process completes, the
    process samples another task from its task sampler. All the tasks are
    synchronized (for step and new_task methods).

    @ivar: make_sampler_fn: function which creates a single TaskSampler.
    @ivar: sampler_fn_args: sequence of dictionaries describing the args
        to pass to make_sampler_fn on each individual process.
    @ivar: auto_resample_when_done: automatically sample a new Task from the TaskSampler when
        the Task completes. If False, a new Task will not be resampled until all
        Tasks on all processes have completed. This functionality is provided for seamless training
        of vectorized Tasks.
    @ivar: multiprocessing_start_method: the multiprocessing method used to
        spawn worker processes. Valid methods are
        ``{'spawn', 'forkserver', 'fork'}`` ``'forkserver'`` is the
        recommended method as it works well with CUDA. If
        ``'fork'`` is used, the subproccess  must be started before
        any other GPU useage.
    """

    observation_space: SpaceDict
    metrics_out_queue: mp.Queue
    _workers: List[Union[mp.Process, Thread]]
    _is_waiting: bool
    _num_processes: int
    _auto_resample_when_done: bool
    _mp_ctx: BaseContext
    _connection_read_fns: List[Callable[[], Any]]
    _connection_write_fns: List[Callable[[Any], None]]

    def __init__(
        self,
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args: Sequence[Dict[str, Any]] = None,
        auto_resample_when_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
    ) -> None:

        self._is_waiting = False
        self._is_closed = True

        assert (
            sampler_fn_args is not None and len(sampler_fn_args) > 0
        ), "number of processes to be created should be greater than 0"

        self._num_processes = len(sampler_fn_args)

        assert multiprocessing_start_method in self._valid_start_methods, (
            "multiprocessing_start_method must be one of {}. Got '{}'"
        ).format(self._valid_start_methods, multiprocessing_start_method)
        self._auto_resample_when_done = auto_resample_when_done
        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self.metrics_out_queue = self._mp_ctx.Queue()
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            make_sampler_fn=make_sampler_fn,
            sampler_fn_args=[
                {"mp_ctx": self._mp_ctx, **args} for args in sampler_fn_args
            ],
        )

        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((OBSERVATION_SPACE_COMMAND, None))

        observation_spaces = [read_fn() for read_fn in self._connection_read_fns]

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
        self.action_spaces = [read_fn() for read_fn in self._connection_read_fns]
        self._paused = []

    @property
    def num_unpaused_tasks(self):
        """
        @return: number of individual unpaused processes.
        """
        return self._num_processes - len(self._paused)

    @property
    def mp_ctx(self):
        """
        @return: multiprocessing context.
        """
        return self._mp_ctx

    @staticmethod
    def _task_sampling_loop_worker(
        worker_id: int,
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args: Dict[str, Any],
        auto_resample_when_done: bool,
        metrics_out_queue: mp.Queue,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        """process worker for creating and interacting with the
        Tasks/TaskSampler."""
        task_sampler = make_sampler_fn(**sampler_fn_args)
        current_task = task_sampler.next_task()

        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    step_result = current_task.step(data)
                    if current_task.is_done():
                        metrics = current_task.metrics()
                        if metrics is not None and len(metrics) != 0:
                            metrics_out_queue.put(metrics)

                        if auto_resample_when_done:
                            current_task = task_sampler.next_task()
                            if current_task is None:
                                step_result = step_result.clone({"observation": None})
                            else:
                                step_result = step_result.clone(
                                    {"observation": current_task.get_observations()}
                                )

                    connection_write_fn(step_result)

                elif command == NEXT_TASK_COMMAND:
                    current_task = task_sampler.next_task()
                    observations = current_task.get_observations()
                    connection_write_fn(observations)

                elif command == RENDER_COMMAND:
                    connection_write_fn(current_task.render(*data[0], **data[1]))

                elif (
                    command == OBSERVATION_SPACE_COMMAND
                    or command == ACTION_SPACE_COMMAND
                ):
                    res = getattr(current_task, command)
                    connection_write_fn(res)

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None or len(function_args) == 0:
                        result = getattr(current_task, function_name)()
                    else:
                        result = getattr(current_task, function_name)(*function_args)
                    connection_write_fn(result)

                elif command == ATTR_COMMAND:
                    property_name = data
                    result = getattr(current_task, property_name)
                    connection_write_fn(result)

                # TODO: update CALL_COMMAND for getting attribute like this
                # elif command == EPISODE_COMMAND:
                #     connection_write_fn(current_task.current_episode)
                elif command == RESET_COMMAND:
                    task_sampler.reset()
                    current_task = task_sampler.next_task()
                    connection_write_fn("done")
                else:
                    raise NotImplementedError()

                command, data = connection_read_fn()

            if child_pipe is not None:
                child_pipe.close()
        except KeyboardInterrupt:
            # logger.info("Worker KeyboardInterrupt")
            print("Worker {} KeyboardInterrupt".format(worker_id))
        finally:
            """Worker {} closing.""".format(worker_id)
            task_sampler.close()

    def _spawn_workers(
        self,
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_connections, worker_connections = zip(
            *[self._mp_ctx.Pipe(duplex=True) for _ in range(self._num_processes)]
        )
        self._workers = []
        # for worker_conn, parent_conn, sampler_fn_args in zip(
        #     worker_connections, parent_connections, sampler_fn_args
        # ):
        for id, stuff in enumerate(
            zip(worker_connections, parent_connections, sampler_fn_args)
        ):
            worker_conn, parent_conn, sampler_fn_args = stuff
            ps = self._mp_ctx.Process(
                target=self._task_sampling_loop_worker,
                args=(
                    id,
                    worker_conn.recv,
                    worker_conn.send,
                    make_sampler_fn,
                    sampler_fn_args,
                    self._auto_resample_when_done,
                    self.metrics_out_queue,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(ps)
            ps.daemon = True
            ps.start()
            worker_conn.close()
        return (
            [p.recv for p in parent_connections],
            [p.send for p in parent_connections],
        )

    # def current_episodes(self):
    #     self._is_waiting = True
    #     for write_fn in self._connection_write_fns:
    #         write_fn((EPISODE_COMMAND, None))
    #     results = []
    #     for read_fn in self._connection_read_fns:
    #         results.append(read_fn())
    #     self._is_waiting = False
    #     return results

    def next_task(self):
        """Move to the the next Task for all TaskSamplers.

        @return: list of initial observations for each of the new tasks.
        """
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((NEXT_TASK_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def get_observations(self):
        """Get observations for all unpaused tasks.

        @return: list of observations for each of the unpaused tasks.
        """
        return self.call(["get_observations"] * self.num_unpaused_tasks,)

    def next_task_at(self, index_process: int) -> List[RLStepResult]:
        """Move to the the next Task from the TaskSampler in index_process
        process in the vector.

        @param index_process: index of the process to be reset
        @return: list of length one containing the observations the newly sampled task.
        """
        self._is_waiting = True
        self._connection_write_fns[index_process]((NEXT_TASK_COMMAND, None))
        results = [self._connection_read_fns[index_process]()]
        self._is_waiting = False
        return results

    def step_at(self, index_process: int, action: int) -> [RLStepResult]:
        """Step in the index_process task in the vector.

        @param index_process: index of the process to be reset
        @param action: the action to take
        @return: list containing the output of step method on the task in the indexed process.
        """
        self._is_waiting = True
        self._connection_write_fns[index_process]((STEP_COMMAND, action))
        results = [self._connection_read_fns[index_process]()]
        self._is_waiting = False
        return results

    def async_step(self, actions: List[int]) -> None:
        """Asynchronously step in the vectorized Tasks.

        @param actions: actions to be performed in the vectorized Tasks.
        """
        self._is_waiting = True
        for write_fn, action in zip(self._connection_write_fns, actions):
            write_fn((STEP_COMMAND, action))

    def wait_step(self) -> List[Dict[str, Any]]:
        """Wait until all the asynchronized processes have synchronized."""
        observations = []
        for read_fn in self._connection_read_fns:
            observations.append(read_fn())
        self._is_waiting = False
        return observations

    def step(self, actions: List[int]):
        """Perform actions in the vectorized tasks.

        @param actions: list of size _num_processes containing action to be taken in each task.
        @return: list of outputs from the step method of tasks.
        """
        self.async_step(actions)
        return self.wait_step()

    def reset_all(self):
        """Reset all task samplers to their initial state.
        """
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((RESET_COMMAND, ""))
        for read_fn in self._connection_read_fns:
            read_fn()
        self._is_waiting = False

    def close(self) -> None:
        if self._is_closed:
            return

        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()

        for write_fn in self._connection_write_fns:
            write_fn((CLOSE_COMMAND, None))

        for _, _, write_fn, _ in self._paused:
            write_fn((CLOSE_COMMAND, None))

        for process in self._workers:
            process.join()

        for _, _, _, process in self._paused:
            process.join()

        self._is_closed = True

    def pause_at(self, index: int) -> None:
        """Pauses computation on the Task in process `index` without destroying
        the Task. This is useful for not needing to call steps on all Tasks
        when only some are active (for example during the last samples of
        running eval).

        @param index: which process to pause. All indexes after this
            one will be shifted down by one.
        """
        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()
        read_fn = self._connection_read_fns.pop(index)
        write_fn = self._connection_write_fns.pop(index)
        worker = self._workers.pop(index)
        self._paused.append((index, read_fn, write_fn, worker))

    def resume_all(self) -> None:
        """Resumes any paused processes."""
        for index, read_fn, write_fn, worker in reversed(self._paused):
            self._connection_read_fns.insert(index, read_fn)
            self._connection_write_fns.insert(index, write_fn)
            self._workers.insert(index, worker)
        self._paused = []

    def call_at(
        self, index: int, function_name: str, function_args: Optional[List[Any]] = None
    ) -> Any:
        """Calls a function (which is passed by name) on the selected task and
        returns the result.

        @param index: which task to call the function on.
        @param function_name: the name of the function to call on the task.
        @param function_args: optional function args.
        @return: result of calling the function.
        """
        self._is_waiting = True
        self._connection_write_fns[index](
            (CALL_COMMAND, (function_name, function_args))
        )
        result = self._connection_read_fns[index]()
        self._is_waiting = False
        return result

    def call(
        self,
        function_names: Union[str, List[str]],
        function_args_list: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Calls a list of functions (which are passed by name) on the
        corresponding task (by index).

        @param function_names: the name of the functions to call on the tasks.
        @param function_args_list: list of function args for each function.
            If provided, len(function_args_list) should be as long as  len(function_names).
        @return: list of results of calling the functions.
        """
        self._is_waiting = True

        if isinstance(function_names, str):
            function_names = [function_names] * self.num_unpaused_tasks

        if function_args_list is None:
            function_args_list = [None] * len(function_names)
        assert len(function_names) == len(function_args_list)
        func_args = zip(function_names, function_args_list)
        for write_fn, func_args_on in zip(self._connection_write_fns, func_args):
            write_fn((CALL_COMMAND, func_args_on))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def attr_at(self, index: int, attr_name: str) -> Any:
        """Gets the attribute (specified by name) on the selected task and
        returns it.

        @param index: which task to call the function on.
        @param attr_name: the name of the function to call on the task.
        @return: result of calling the function.
        """
        self._is_waiting = True
        self._connection_write_fns[index]((ATTR_COMMAND, attr_name))
        result = self._connection_read_fns[index]()
        self._is_waiting = False
        return result

    def attr(self, attr_names: Union[List[str], str]) -> List[Any]:
        """Gets the attributes (specified by name) on the tasks.

        @param attr_names: the name of the functions to call on the tasks.
        @param function_args_list: list of function args for each function.
            If provided, len(function_args_list) should be as long as  len(function_names).
        @return: list of results of calling the functions.
        """
        self._is_waiting = True

        if isinstance(attr_names, str):
            attr_names = [attr_names] * self.num_unpaused_tasks

        for write_fn, attr_name in zip(self._connection_write_fns, attr_names):
            write_fn((ATTR_COMMAND, attr_name))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def render(self, mode: str = "human", *args, **kwargs) -> Union[np.ndarray, None]:
        """Render observations from all Tasks in a tiled image."""
        for write_fn in self._connection_write_fns:
            write_fn((RENDER_COMMAND, (args, {"mode": "rgb", **kwargs})))
        images = [read_fn() for read_fn in self._connection_read_fns]
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


class ThreadedVectorSampledTasks(VectorSampledTasks):
    """Provides same functionality as ``VectorSampledTasks``, the only
    difference is it runs in a multi-thread setup inside a single process.

    ``VectorSampledTasks`` runs in a multi-proc setup. This makes it
    much easier to debug when using ``VectorSampledTasks`` because you
    can actually put break points in the Task methods. It should not be
    used for best performance.
    """

    def _spawn_workers(
        self,
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_read_queues, parent_write_queues = zip(
            *[(Queue(), Queue()) for _ in range(self._num_processes)]
        )
        self._workers = []
        for id, stuff in enumerate(
            zip(parent_read_queues, parent_write_queues, sampler_fn_args)
        ):
            parent_read_queue, parent_write_queue, sampler_fn_args = stuff
            thread = Thread(
                target=self._task_sampling_loop_worker,
                args=(
                    parent_write_queue.get,
                    parent_read_queue.put,
                    make_sampler_fn,
                    sampler_fn_args,
                    self._auto_resample_when_done,
                    self.metrics_out_queue,
                ),
            )
            self._workers.append(thread)
            thread.daemon = True
            thread.start()
        return (
            [q.get for q in parent_read_queues],
            [q.put for q in parent_write_queues],
        )
