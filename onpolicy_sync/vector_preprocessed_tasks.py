#!/usr/bin/env python3

# Original work Copyright (c) Facebook, Inc. and its affiliates.
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import typing
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from threading import Thread
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple, Union, Dict

import numpy as np
import torch.multiprocessing as mp
from gym.spaces.dict import Dict as SpaceDict
from setproctitle import setproctitle as ptitle

from onpolicy_sync.vector_sampled_tasks import VectorSampledTasks
from rl_base.common import RLStepResult
from rl_base.preprocessor import ObservationSet
from rl_base.task import TaskSampler
from utils.system import init_logging, LOGGER
from utils.tensor_utils import tile_images, batch_observations

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

# EPISODE_COMMAND = "current_episode"
RESET_COMMAND = "reset"
SEED_COMMAND = "seed"

PAUSE_COMMAND = "pause"
RESUME_COMMAND = "resume"
CALL_AT_COMMAND = "call_at"


class VectorPreprocessedTasks(object):
    """Vectorized collection of preprocessed tasks. Creates multiple processes
    where each process creates its owe own preprocessor and and
    VectorSampledTasks. This class allows for interacting with all preprocessed
    tasks in a vectorized manner, identically to the functionality offered by
    VectorSampledTasks.

    # Attributes

    make_preprocessors_fn : sequence of functions to create an ObservationSet for a process managing a group of samplers
    task_sampler_ids : sequence of sequences of worker indices for each manager process
    make_sampler_fn : function which creates a single TaskSampler.
    sampler_fn_args : sequence of dictionaries describing the args
        to pass to make_sampler_fn on each individual process in VectorSampledTasks.
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
        make_preprocessors_fn: Sequence[Callable[..., ObservationSet]],
        task_sampler_ids: Sequence[Sequence[int]],
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args: Sequence[Dict[str, Any]] = None,
        auto_resample_when_done: bool = True,
        multiprocessing_start_method: Optional[str] = "forkserver",
        mp_ctx: Optional[BaseContext] = None,
        metrics_out_queue: mp.Queue = None,
    ) -> None:

        self._is_waiting = False
        self._is_closed = True

        assert (
            sampler_fn_args is not None and len(sampler_fn_args) > 0
        ), "number of processes to be created should be greater than 0"

        self._num_processes = len(sampler_fn_args)
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
            self._mp_ctx = typing.cast(BaseContext, mp_ctx)
        self.metrics_out_queue = metrics_out_queue or self._mp_ctx.Queue()
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            make_preprocessors_fn=make_preprocessors_fn,
            task_sampler_ids=task_sampler_ids,
            make_sampler_fn=make_sampler_fn,
            sampler_fn_args=[
                {"mp_ctx": self._mp_ctx, **args} for args in sampler_fn_args
            ],
        )

        self.manager_to_task_sampler = task_sampler_ids
        self.task_sampler_to_manager = {}
        count = 0
        for mit in range(len(self.manager_to_task_sampler)):
            for wit in range(len(self.manager_to_task_sampler[mit])):
                self.task_sampler_to_manager[count] = mit
                count += 1

        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((OBSERVATION_SPACE_COMMAND, None))

        observation_spaces = []
        for read_fn in self._connection_read_fns:
            observation_spaces += read_fn()

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

        self.action_spaces = []
        for read_fn in self._connection_read_fns:
            self.action_spaces += read_fn()

        self._original_id: List[int] = list(
            range(self._num_processes)
        )  # when pause, pop item from _original_id

    @property
    def num_unpaused_tasks(self) -> int:
        """Number of unpaused processes.

        # Returns

        Number of unpaused processes.
        """
        return len(self._original_id)

    @property
    def mp_ctx(self):
        """Get the multiprocessing process used by the vector task.

        # Returns

        The multiprocessing context.
        """
        return self._mp_ctx

    @staticmethod
    def _task_sampling_loop_worker(
        worker_id: int,
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        make_preprocessors_fn: Callable[..., ObservationSet],
        task_sampler_ids: Sequence[int],
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args: Sequence[Dict[str, Any]],
        auto_resample_when_done: bool,
        metrics_out_queue: mp.Queue,
        mp_ctx: BaseContext,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
        vectorized_class: Any = VectorSampledTasks,
    ) -> None:
        """process worker for creating and interacting with the
        Tasks/TaskSampler."""
        ptitle("VectorPreprocessedTasks: {}".format(worker_id))

        init_logging()

        observation_set = make_preprocessors_fn(worker_id)
        vector_task_sampler = vectorized_class(
            make_sampler_fn=make_sampler_fn,
            sampler_fn_args=[sampler_fn_args[it] for it in task_sampler_ids],
            multiprocessing_start_method=None,
            mp_ctx=mp_ctx,
            auto_resample_when_done=auto_resample_when_done,
            metrics_out_queue=metrics_out_queue,
        )

        def update_observations(observations):
            def _remove_paused(observations):
                paused, keep, running = [], [], []
                for it, obs in enumerate(observations):
                    if obs is None:
                        paused.append(it)
                    else:
                        keep.append(it)
                        running.append(obs)

                batch = batch_observations(running, device=observation_set.device)

                return len(paused), keep, batch

            def unbatch_observations(batch, bsize):
                def traverse_batch(batch, res, it):
                    for obs_name in batch:
                        if isinstance(batch[obs_name], Dict):
                            res[obs_name] = {}
                            traverse_batch(batch[obs_name], res[obs_name], it)
                        else:
                            res[obs_name] = batch[obs_name][it].cpu()

                res = []
                for it in range(bsize):
                    res.append({})
                    traverse_batch(batch, res[-1], it)
                return res

            npaused, keep, batched_observations = _remove_paused(observations)
            if len(keep) > 0:
                batched_observations = observation_set.get_observations(
                    batched_observations
                )
                unbatched_obs = unbatch_observations(batched_observations, len(keep))
                for it, pos in enumerate(keep):
                    observations[pos] = unbatched_obs[it]
            return observations

        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    if vector_task_sampler.num_unpaused_tasks == 0:
                        connection_write_fn([])  # empty list
                        assert (
                            len(data[1]) == 0
                        ), "Passing actions to manager with 0 unpaused tasks"
                    else:
                        if data[0] == -1:
                            step_result = vector_task_sampler.step(
                                data[1]
                            )  # data[1] is a list of actions for the workers of this manager
                        else:
                            step_result = vector_task_sampler.step_at(
                                data[0], data[1]
                            )  # data[1] is an int (action for specific worker)

                        observations, rewards, dones, infos = [
                            list(x) for x in zip(*step_result)
                        ]
                        observations = update_observations(observations)

                        step_result = [
                            RLStepResult(obs, reward, done, info)
                            for obs, reward, done, info in zip(
                                observations, rewards, dones, infos
                            )
                        ]

                        connection_write_fn(
                            step_result
                        )  # step result is a list of results

                elif command == NEXT_TASK_COMMAND:
                    if vector_task_sampler.num_unpaused_tasks == 0:
                        connection_write_fn([])  # empty list
                    else:
                        if data[1] is not None:
                            if data[0] == -1:
                                vector_task_sampler.next_task(
                                    **data
                                )  # data is a kwargs shared by all subworkers
                            else:
                                LOGGER.error(
                                    "passing arguments to directed next_task (worker {}, args {})".format(
                                        data[0], data[1]
                                    )
                                )
                        else:
                            if data[0] == -1:
                                vector_task_sampler.next_task()
                            else:
                                vector_task_sampler.next_task_at(data[0])

                        observations = vector_task_sampler.get_observations()
                        observations = update_observations(observations)

                        connection_write_fn(observations)  # list of observations

                elif command == RENDER_COMMAND:
                    if vector_task_sampler.num_unpaused_tasks == 0:
                        connection_write_fn([])  # empty list
                    else:
                        # TODO support other modes, never using human!
                        # same args and kwargs for all workers
                        res = vector_task_sampler.render(
                            mode="rgb_list", *data[0], **data[1]
                        )
                        connection_write_fn(res)  # res is a list
                elif (
                    command == OBSERVATION_SPACE_COMMAND
                    or command == ACTION_SPACE_COMMAND
                ):
                    for write_fn in vector_task_sampler._connection_write_fns:
                        write_fn((command, None))

                    res = [
                        read_fn()
                        for read_fn in vector_task_sampler._connection_read_fns
                    ]

                    connection_write_fn(res)

                elif command == CALL_COMMAND:
                    if vector_task_sampler.num_unpaused_tasks == 0:
                        connection_write_fn([])  # empty list
                    else:
                        if data[0] == -1:
                            function_names, function_args = data[1]
                            results = vector_task_sampler.call(
                                function_names, function_args
                            )
                            connection_write_fn(
                                results
                            )  # list of results for all active workers
                        else:
                            function_name, function_args = data[1]
                            result = vector_task_sampler.call_at(
                                data[0], function_name, function_args
                            )
                            connection_write_fn([result])  # result is not a list

                elif command == SAMPLER_COMMAND:
                    if vector_task_sampler.num_unpaused_tasks == 0:
                        connection_write_fn([])  # empty list
                    else:
                        if data[0] == -1:
                            function_names, function_args = data[1]
                            results = vector_task_sampler.call(
                                function_names, function_args, call_sampler=True
                            )
                            connection_write_fn(
                                results
                            )  # list of results for all active workers
                        else:
                            function_name, function_args = data[1]
                            result = vector_task_sampler.call_at(
                                data[0], function_name, function_args, call_sampler=True
                            )
                            connection_write_fn([result])  # result is not a list

                elif command == ATTR_COMMAND:
                    if vector_task_sampler.num_unpaused_tasks == 0:
                        connection_write_fn([])  # empty list
                    else:
                        if data[0] == -1:
                            property_name = data[1]
                            results = vector_task_sampler.attr(property_name)
                            connection_write_fn(
                                results
                            )  # list of results for all active workers
                        else:
                            property_name = data[1]
                            result = vector_task_sampler.attr_at(data[0], property_name)
                            connection_write_fn([result])  # result is not a list

                elif command == SAMPLER_ATTR_COMMAND:
                    if vector_task_sampler.num_unpaused_tasks == 0:
                        connection_write_fn([])  # empty list
                    else:
                        if data[0] == -1:
                            property_name = data[1]
                            results = vector_task_sampler.attr(
                                property_name, call_sampler=True
                            )
                            connection_write_fn(
                                results
                            )  # list of results for all active workers
                        else:
                            property_name = data[1]
                            result = vector_task_sampler.attr_at(
                                data[0], property_name, call_sampler=True
                            )
                            connection_write_fn([result])  # result is not a list

                elif command == RESET_COMMAND:
                    vector_task_sampler.reset_all()  # already sets a new task
                    connection_write_fn("done")
                elif command == SEED_COMMAND:
                    vector_task_sampler.set_seeds(data)  # one for each subworker
                    connection_write_fn("done")
                elif command == PAUSE_COMMAND:
                    vector_task_sampler.pause_at(data)
                    connection_write_fn("done")
                elif command == RESUME_COMMAND:
                    vector_task_sampler.resume_all()
                    connection_write_fn("done")
                else:
                    raise NotImplementedError()

                command, data = connection_read_fn()

            if child_pipe is not None:
                child_pipe.close()
        except KeyboardInterrupt:
            print("Manager {} KeyboardInterrupt".format(worker_id))
        finally:
            """Manager {} closing.""".format(worker_id)
            vector_task_sampler.close()

    def _spawn_workers(
        self,
        make_preprocessors_fn: Sequence[Callable[..., ObservationSet]],
        task_sampler_ids: Sequence[Sequence[int]],
        make_sampler_fn: Callable[..., TaskSampler],
        sampler_fn_args: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_connections, worker_connections = zip(
            *[self._mp_ctx.Pipe(duplex=True) for _ in range(self._num_processes)]
        )
        self._workers = []
        for id, stuff in enumerate(
            zip(
                worker_connections,
                parent_connections,
                make_preprocessors_fn,
                task_sampler_ids,
            )
        ):
            worker_conn, parent_conn, cur_make_preprocessors_fn, cur_task_sampler_ids = stuff  # type: ignore
            LOGGER.info("Starting {}-th preprocessor manager".format(id))
            ps = self._mp_ctx.Process(  # type: ignore
                target=self._task_sampling_loop_worker,
                args=(
                    id,
                    worker_conn.recv,
                    worker_conn.send,
                    cur_make_preprocessors_fn,
                    cur_task_sampler_ids,
                    make_sampler_fn,
                    sampler_fn_args,
                    self._auto_resample_when_done,
                    self.metrics_out_queue,
                    self._mp_ctx,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(ps)
            ps.daemon = False
            ps.start()
            worker_conn.close()
            time.sleep(0.1)  # Useful to reduce the level of rush in our modern life
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
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((NEXT_TASK_COMMAND, (-1, kwargs)))
        results = []
        for read_fn in self._connection_read_fns:
            results += read_fn()  # concatenate all lists from each manager
        self._is_waiting = False
        return results

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

        index_process : Index of the process to be reset.

        # Returns

        List of length one containing the observations the newly sampled task.
        """
        self._is_waiting = True
        manager = self.manager(index_process)
        self._connection_write_fns[manager](
            (NEXT_TASK_COMMAND, (self.manager_worker(index_process), None))
        )
        results = self._connection_read_fns[manager]()  # it's already a list
        self._is_waiting = False
        return results

    def step_at(self, index_process: int, action: int) -> List[RLStepResult]:
        """Step in the index_process task in the vector.

        # Parameters

        index_process : Index of the process to be reset.
        action : The action to take.

        # Returns

        List containing the output of step method on the task in the indexed process.
        """
        self._is_waiting = True
        manager = self.manager(index_process)
        self._connection_write_fns[manager](
            (STEP_COMMAND, (self.manager_worker(index_process), action))
        )
        results = self._connection_read_fns[manager]()  # it's already a list
        self._is_waiting = False
        return results

    def async_step(self, action_lists: List[List[int]]) -> None:
        """Asynchronously step in the vectorized Tasks.

        # Parameters

        actions : actions to be performed in the vectorized Tasks.
        """
        self._is_waiting = True
        for write_fn, action_list in zip(self._connection_write_fns, action_lists):
            write_fn((STEP_COMMAND, (-1, action_list)))

    def wait_step(self) -> List[Dict[str, Any]]:
        """Wait until all the asynchronized processes have synchronized."""
        observations = []
        for read_fn in self._connection_read_fns:
            observations += read_fn()  # it's already a list
        self._is_waiting = False
        return observations

    def step(self, actions: List[int]):
        """Perform actions in the vectorized tasks.

        # Parameters

        actions: List of size _num_processes containing action to be taken in each task.

        # Returns

        List of outputs from the step method of tasks.
        """

        action_lists = [
            [] * len(self.manager_to_task_sampler)
        ]  # empty action list for each manager
        assert len(action_lists) == len(
            self._connection_write_fns
        ), "Mismatch between number of managers and pipes"

        for it, action in enumerate(actions):
            action_lists[self.manager(it)].append(action)

        self.async_step(action_lists)
        return self.wait_step()

    def reset_all(self):
        """Reset all task samplers to their initial state (except for the RNG
        seed)."""
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((RESET_COMMAND, ""))
        for read_fn in self._connection_read_fns:
            read_fn()
        self._is_waiting = False

    def set_seeds(self, seeds: List[int]):
        """Sets new tasks' RNG seeds.

        # Parameters

        seeds: List of size _num_processes containing new RNG seeds.
        """
        self._is_waiting = True
        for write_fn, seed in zip(self._connection_write_fns, seeds):
            write_fn((SEED_COMMAND, seed))
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

        for process in self._workers:
            process.join()

        self._is_closed = True

    def manager(self, index: int) -> int:
        return self.task_sampler_to_manager[self._original_id[index]]

    def manager_worker(self, index):
        original_workers_in_manager = self.manager_to_task_sampler[self.manager(index)]
        original_worker_index = self._original_id[index]
        running_ids = set(self._original_id)

        pos_in_manager = 0
        for pos in original_workers_in_manager:
            if pos == original_worker_index:
                # assert pos in running_ids, "accessing paused worker"  # unnecessary (original_worker_index always fulfills it)
                break
            if pos in running_ids:
                pos_in_manager += 1

        return pos_in_manager

    def pause_at(self, index: int) -> None:
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

        self._is_waiting = True
        manager = self.manager(index)
        self._connection_write_fns[manager]((PAUSE_COMMAND, self.manager_worker(index)))
        self._connection_read_fns[manager]()
        self._is_waiting = False

        self._original_id.pop(index)

    def resume_all(self) -> None:
        """Resumes any paused processes."""
        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()

        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((RESUME_COMMAND, ""))
        for read_fn in self._connection_read_fns:
            read_fn()
        self._is_waiting = False

        self._original_id = list(range(self._num_processes))

    def call_at(
        self,
        index: int,
        function_name: str,
        function_args: Optional[List[Any]] = None,
        call_sampler: bool = False,
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
        self._is_waiting = True
        manager = self.manager(index)
        self._connection_write_fns[manager](
            (
                CALL_COMMAND if not call_sampler else SAMPLER_COMMAND,
                (self.manager_worker(index), (function_name, function_args)),
            )
        )
        result = self._connection_read_fns[index]()[0]  # it's a list
        self._is_waiting = False
        return result

    def call(
        self,
        function_names: Union[str, List[str]],
        function_args_list: Optional[List[Any]] = None,
        call_sampler: bool = False,
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
        func_args = zip(function_names, function_args_list)
        for write_fn, func_args_on in zip(self._connection_write_fns, func_args):
            write_fn(
                (
                    CALL_COMMAND if not call_sampler else SAMPLER_COMMAND,
                    (-1, func_args_on),
                )
            )
        results = []
        for read_fn in self._connection_read_fns:
            results += read_fn()  # it already returns list
        self._is_waiting = False
        return results

    def attr_at(self, index: int, attr_name: str, call_sampler: bool = False) -> Any:
        """Gets the attribute (specified by name) on the selected task and
        returns it.

        # Parameters

        index : Which task to call the function on.
        attr_name : The name of the function to call on the task.

        # Returns

         Result of calling the function.
        """
        self._is_waiting = True
        manager = self.manager(index)
        self._connection_write_fns[manager](
            (
                ATTR_COMMAND if not call_sampler else SAMPLER_ATTR_COMMAND,
                (self.manager_worker(index), attr_name),
            )
        )
        result = self._connection_read_fns[index]()[0]  # already a list
        self._is_waiting = False
        return result

    def attr(self, attr_names: Union[List[str], str], call_sampler=False) -> List[Any]:
        """Gets the attributes (specified by name) on the tasks.

        # Parameters

        attr_names : The name of the functions to call on the tasks.

        # Returns

        List of results of calling the functions.
        """
        self._is_waiting = True

        if isinstance(attr_names, str):
            attr_names = [attr_names] * self.num_unpaused_tasks

        for write_fn, attr_name in zip(self._connection_write_fns, attr_names):
            write_fn(
                (
                    ATTR_COMMAND if not call_sampler else SAMPLER_ATTR_COMMAND,
                    (-1, attr_name),
                )
            )
        results = []
        for read_fn in self._connection_read_fns:
            results += read_fn()  # it's already a list
        self._is_waiting = False
        return results

    def render(
        self, mode: str = "human", *args, **kwargs
    ) -> Union[np.ndarray, None, List[np.ndarray]]:
        """Render observations from all Tasks in a tiled image."""
        for write_fn in self._connection_write_fns:
            write_fn((RENDER_COMMAND, (args, {"mode": "rgb_list", **kwargs})))
        manager_lists = [read_fn() for read_fn in self._connection_read_fns]

        valid_image = None
        for valid_list in manager_lists:
            if len(valid_list) > 0:
                valid_image = valid_list[0]
                break

        assert valid_image is not None, "No valid image found"

        flat_list: List[np.ndarray] = []
        for it in range(len(self.manager_to_task_sampler)):
            if len(manager_lists[it]) == 0:
                for wit in range(len(self.manager_to_task_sampler[it])):
                    flat_list.append(np.zeros_like(valid_image))
            else:
                assert len(manager_lists[it]) == len(
                    self.manager_to_task_sampler[it]
                ), "number of rendered images {} not matching number of workers {}".format(
                    len(manager_lists[it]), len(self.manager_to_task_sampler[it])
                )
                flat_list += manager_lists[it]

        if mode == "rgb_list":
            return flat_list

        tile = tile_images(flat_list)
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
