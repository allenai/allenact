from typing import Dict, Any, Union, Optional, List, Tuple, Sequence
import abc
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from utils.tensor_utils import SummaryWriter, tile_images, process_video
from utils.experiment_utils import Builder
from utils.system import LOGGER
from rl_base.common import ActorCriticOutput


class AbstractViz:
    def __init__(
            self,
            label: Optional[str] = None,
            vector_task_sources: Sequence[Tuple[str, Dict[str, Any]]] = (),
            rollout_sources: Sequence[Tuple[str, Sequence[str], int]] = (),
            actor_critic_source: bool = False,
    ):
        self.label = label
        self.vector_task_sources = list(vector_task_sources)
        self.rollout_sources = list(rollout_sources)
        self.actor_critic_source = actor_critic_source

        self.mode: Optional[str] = None
        self.path_to_id: Optional[Sequence[str]] = None
        self.episode_ids: Optional[Sequence[Sequence[str]]] = None

    @staticmethod
    def source_to_str(source, is_vector_task):
        source_type = "vector_task" if is_vector_task else "rollout"
        return "__{}_sep__".format(source_type).join(["{}".format(s) for s in source])

    @staticmethod
    def access(dictionary, path):
        path = path[::-1]
        while len(path) > 0:
            dictionary = dictionary[path.pop()]
        return dictionary

    def setup(
            self,
            mode: str,
            path_to_id: Sequence[str],
            episode_ids: Sequence[Union[Sequence[str], str]],
            force: bool = False,
    ):
        self.mode = mode
        self.path_to_id = list(path_to_id)
        if self.episode_ids is None or force:
            self.episode_ids = list(episode_ids) if not isinstance(episode_ids[0], str) else [list(episode_ids)]

    @abc.abstractmethod
    def log(
            self,
            log_writer: SummaryWriter,
            task_outputs: Optional[List[Any]],
            render: Optional[Dict[str, List[Dict[str, Any]]]],
            num_steps: int,
    ):
        raise NotImplementedError


class TrajectoryViz(AbstractViz):
    def __init__(
            self,
            path_to_trajectory: Sequence[str] = ("task_info", "followed_path"),
            path_to_target_location: Optional[Sequence[str]] = ("task_info", "target_position"),
            x: str = "x",
            y: str = "z",
            label: str = "trajectories",
            figsize: Tuple[int, int] = (2, 2),
            fontsize: int = 5,
    ):
        super().__init__(label)
        self.x = x
        self.y = y
        self.path_to_trajectory = list(path_to_trajectory)
        self.path_to_target_location = list(path_to_target_location) if path_to_target_location is not None else None
        self.figsize = figsize
        self.fontsize = fontsize

    def log(
            self,
            log_writer: SummaryWriter,
            task_outputs: Optional[List[Any]],
            render: Optional[Dict[str, List[Dict[str, Any]]]],
            num_steps: int,
    ):
        if task_outputs is None:
            return

        all_episodes = {self.access(episode, self.path_to_id): episode for episode in task_outputs}

        for page, current_ids in enumerate(self.episode_ids):
            figs = []
            for episode_id in current_ids:
                assert episode_id in all_episodes
                figs.append(self.make_fig(all_episodes[episode_id], episode_id))
            log_writer.add_figure("{}/{}_group{}".format(self.mode, self.label, page), figs, global_step=num_steps)
            plt.close("all")  # close all current figures (SummaryWriter already closes all figures we log)

    def make_fig(self, episode, episode_id):
        # From https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
        def colorline(x, y, z=None, cmap=plt.get_cmap('cool'), norm=plt.Normalize(0.0, 1.0), linewidth=2, alpha=1.0):
            """
            Plot a colored line with coordinates x and y
            Optionally specify colors in the array z
            Optionally specify a colormap, a norm function and a line width
            """

            def make_segments(x, y):
                """
                Create list of line segments from x and y coordinates, in the correct format for LineCollection:
                an array of the form   numlines x (points per line) x 2 (x and y) array
                """
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                return segments

            # Default colors equally spaced on [0,1]:
            if z is None:
                z = np.linspace(0.0, 1.0, len(x))

            # Special case if a single number:
            if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
                z = np.array([z])

            z = np.asarray(z)

            segments = make_segments(x, y)
            lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)

            ax = plt.gca()
            ax.add_collection(lc)

            return lc

        trajectory = self.access(episode, self.path_to_trajectory)

        x, y = [], []
        for xy in trajectory:
            x.append(xy[self.x])
            y.append(xy[self.y])

        fig, ax = plt.subplots(figsize=self.figsize)
        colorline(x, y)
        ax.scatter([x[0]], [y[0]], marker=">")  # play
        ax.scatter([x[-1]], [y[-1]], marker="s")  # stop

        if self.path_to_target_location is not None:
            target = self.access(episode, self.path_to_target_location)
            ax.scatter([target[self.x]], [target[self.y]], marker="*")

        ax.set_title(episode_id, fontsize=self.fontsize)
        ax.tick_params(axis='x', labelsize=self.fontsize)
        ax.tick_params(axis='y', labelsize=self.fontsize)

        return fig


class AgentViewViz(AbstractViz):
    def __init__(
            self,
            label: str = "agent_view",
            max_clip_length: int = 100,  # control memory used when converting groups of images into clips
            max_video_length: int = -1,  # no limit, if > 0, limit the maximum video length (discard last frames)
            vector_task_source: Tuple[str, Dict[str, Any]] = ("render", {"mode": "raw_rgb_list"}),
    ):
        super().__init__(label, vector_task_sources=[vector_task_source])
        self.max_clip_length = max_clip_length
        self.max_video_length = max_video_length

    def log(
            self,
            log_writer: SummaryWriter,
            task_outputs: Optional[List[Any]],
            render: Optional[Dict[str, List[Dict[str, Any]]]],
            num_steps: int,
    ):
        if render is None:
            return

        datum_id = self.source_to_str(self.vector_task_sources[0], is_vector_task=True)
        for page, current_ids in enumerate(self.episode_ids):
            images = []  # list of lists of rgb frames
            for episode_id in current_ids:
                assert episode_id in render
                images.append([step[datum_id] for step in render[episode_id]])
                # TODO overlay episode id
            vid = self.make_vid(images)
            if vid is not None:
                log_writer.add_vid("{}/{}_group{}".format(self.mode, self.label, page), vid, global_step=num_steps)

    def make_vid(self, images):
        max_length = max([len(ep) for ep in images])

        if max_length == 0:
            return None

        valid_im = None
        for ep in images:
            if len(ep) > 0:
                valid_im = ep[0]
                break

        frames = []
        for it in range(max_length):
            current_images = []
            for ep in images:
                if it < len(ep):
                    current_images.append(ep[it])
                else:
                    if it == 0:
                        current_images.append(np.zeros_like(valid_im))
                    else:
                        gray = ep[-1].copy()
                        gray[:, :, 0] = gray[:, :, 2] = gray[:, :, 1]
                        current_images.append(gray)
            frames.append(tile_images(current_images))

        return process_video(frames, self.max_clip_length, self.max_video_length)


# class TensorViz(AbstractViz):
#     def __init__(
#             self,
#             label: str = "action_probs",
#             rollout_source: Tuple[str, Sequence[str], int] = ("action_log_probs", [], 1),
#             softmax_dim: Optional[int] = -1,
#             figsize: Tuple[int, int] = (3, 3),
#     ):
#         super().__init__(label, rollout_sources=[rollout_source])
#         self.softmax_dim = softmax_dim
#         self.figsize = figsize
#         self.datum_id = self.source_to_str(self.rollout_sources[0], is_vector_task=False)
#
#     def log(
#             self,
#             log_writer: SummaryWriter,
#             task_outputs: Optional[List[Any]],
#             render: Optional[Dict[str, List[Dict[str, Any]]]],
#             num_steps: int,
#     ):
#         if render is None:
#             return
#
#         for page, current_ids in enumerate(self.episode_ids):
#             figs = []
#             for episode_id in current_ids:
#                 assert episode_id in render
#                 # TODO store as list. there should never be a missing step except for sometimes at the end
#                 episode_src = {
#                     it: step[self.datum_id] for it, step in enumerate(render[episode_id]) if self.datum_id in step
#                 }
#                 figs.append(self.make_fig(episode_src, episode_id))
#             log_writer.add_figure("{}/{}_group{}".format(self.mode, self.label, page), figs, global_step=num_steps)
#             plt.close("all")  # close all current figures (SummaryWriter already closes all figures we log)
#
#     def make_fig(self, episode_src, episode_id):
#         all_steps = sorted(list(episode_src.keys()))
#
#         # TODO no need to fill missing steps with zeros? They should only be missing at the end...
#         LOGGER.debug("{} collected steps for {}: {}".format(len(all_steps), episode_id, all_steps))
#         # assert all_steps[-1] == len(all_steps) - 1
#
#         mats = []
#         for step in all_steps:
#             LOGGER.debug("{} {} {}".format(episode_id, step, episode_src[step].shape))
#             mats.append(episode_src[step])
#
#         # Concatenate along step axis (0)
#         mat = np.concatenate(mats, axis=0)
#
#         # Convert to probabilities
#         mat = self.softmax(mat)
#
#         # Remove episode/sampler axis
#         mat = mat.squeeze(axis=self.rollout_sources[0][2])
#
#         fig, ax = plt.subplots(figsize=self.figsize)
#         ax.matshow(mat)
#         return fig
#
#     def softmax(self, x):
#         if self.softmax_dim is not None:
#             ex = np.exp(x - np.max(x, axis=self.softmax_dim, keepdims=True))
#             return ex / ex.sum(axis=self.softmax_dim, keepdims=True)
#         return x

class ActorViz(AbstractViz):
    def __init__(
            self,
            label: str = "action_probs",
            action_names_path: Optional[Sequence[str]] = ("task_info", "action_names"),
            figsize: Tuple[int, int] = (1, 5),
            fontsize: int = 5,
    ):
        super().__init__(label, actor_critic_source=True)
        self.action_names_path = list(action_names_path) if action_names_path is not None else None
        self.figsize = figsize
        self.fontsize = fontsize
        self.action_names = None

    def log(
            self,
            log_writer: SummaryWriter,
            task_outputs: Optional[List[Any]],
            render: Optional[Dict[str, List[Dict[str, Any]]]],
            num_steps: int,
    ):
        if render is None:
            return

        if (
                self.action_names is None
                and task_outputs is not None
                and len(task_outputs) > 0
                and self.action_names_path is not None
        ):
            self.action_names = list(self.access(task_outputs[0], self.action_names_path))

        for page, current_ids in enumerate(self.episode_ids):
            figs = []
            for episode_id in current_ids:
                assert episode_id in render
                # TODO store as list. there should never be a missing step except for sometimes at the end
                episode_src = {
                    it: step["actor_probs"] for it, step in enumerate(render[episode_id]) if "actor_probs" in step
                }
                figs.append(self.make_fig(episode_src, episode_id))
            log_writer.add_figure("{}/{}_group{}".format(self.mode, self.label, page), figs, global_step=num_steps)
            plt.close("all")  # close all current figures (SummaryWriter already closes all figures we log)

    def make_fig(self, episode_src, episode_id):
        all_steps = sorted(list(episode_src.keys()))

        # TODO no need to fill missing steps with zeros? They should only be missing at the end...
        LOGGER.debug("{} collected steps for {}: {}".format(len(all_steps), episode_id, all_steps))
        # assert all_steps[-1] == len(all_steps) - 1

        mats = []
        for step in all_steps:
            LOGGER.debug("{} {} {}".format(episode_id, step, episode_src[step].shape))
            mats.append(episode_src[step])

        # Concatenate along step axis (0, taken from sampler axis)
        mat = np.concatenate(mats, axis=0)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.matshow(mat)

        if self.action_names is not None:
            assert len(self.action_names) == mat.shape[1]
            ax.set_xticklabels([''] + self.action_names, rotation='vertical')

        ax.set_xlabel(episode_id, fontsize=self.fontsize)
        ax.tick_params(axis='x', labelsize=self.fontsize)
        ax.tick_params(axis='y', labelsize=self.fontsize)
        ax.tick_params(bottom=False)

        # Gridlines based on minor ticks
        ax.set_yticks(np.arange(-.5, mat.shape[0], 1), minor=True)
        ax.set_xticks(np.arange(-.5, mat.shape[1], 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.05)
        ax.tick_params(axis='both', which="minor", left=False, top=False, right=False, bottom=False)

        ax.set_aspect('auto')
        plt.tight_layout()
        return fig


class SimpleViz(AbstractViz):
    def __init__(
            self,
            episode_ids: Sequence[Union[Sequence[str], str]],
            path_to_id: Sequence[str] = ("task_info", "id"),
            mode: str = "valid",
            *viz,
            **kw_viz,
    ):
        super().__init__()
        self.setup(mode, path_to_id, episode_ids)

        self.viz = [
            v() if isinstance(v, Builder) else v
            for v in viz if isinstance(v, Builder) or isinstance(v, AbstractViz)
        ] + [
            v() if isinstance(v, Builder) else v
            for k, v in kw_viz.items() if isinstance(v, Builder) or isinstance(v, AbstractViz)
        ]

        self.rollout_sources, self.vector_task_sources, self.actor_critic_source = self._setup_sources()

        self.data = {}  # dict of episode id to list of dicts with collected data
        self.last_it2epid = []

        self.all_episode_ids = self._episodes_set()

    def _setup_sources(self):
        rollout_sources, vector_task_sources = [], []
        labels = []
        actor_critic_source = False
        for v in self.viz:
            v.setup(self.mode, self.path_to_id, self.episode_ids)
            labels.append(v.label)
            rollout_sources += v.rollout_sources
            vector_task_sources += v.vector_task_sources
            actor_critic_source |= v.actor_critic_source
        LOGGER.info("Logging labels {}".format(labels))

        LOGGER.debug("rollout sources {}".format(rollout_sources))
        LOGGER.debug("vector task sources {}".format(vector_task_sources))
        LOGGER.debug("actor-critic source {}".format(actor_critic_source))

        rol_flat = {json.dumps(src, sort_keys=True): src for src in rollout_sources}
        vt_flat = {json.dumps(src, sort_keys=True): src for src in vector_task_sources}

        rol_keys = list(set(rol_flat.keys()))
        vt_keys = list(set(vt_flat.keys()))

        return [rol_flat[k] for k in rol_keys], [vt_flat[k] for k in vt_keys], actor_critic_source

    def _episodes_set(self):
        all_episode_ids = []
        for group in self.episode_ids:
            all_episode_ids += group
        return set(all_episode_ids)

    def empty(self):
        return len(self.data) == 0

    # to be called by engine
    def collect(self, vector_task, alive, rollout=None, actor_critic=None):
        # TODO we miss tensors for the last step in the last episode of each sampler
        # TODO assume we never revisit same episode? we have one entry per episode id in data
        LOGGER.debug("Data entries: {}".format(list(self.data.keys())))
        for entry in self.data:
            LOGGER.debug("{}: {} steps, last {}".format(entry, len(self.data[entry]), list(self.data[entry][-1].keys())))

        # 1. find the identifiers of current episodes through vector_task
        infos = vector_task.attr("task_info")
        it2epid = [self.access(info, self.path_to_id[1:]) for info in infos]

        assert len(alive) == len(it2epid)
        assert len(alive) <= len(self.last_it2epid) or len(self.last_it2epid) == 0

        # 2. gather vector_task_sources (in phase with it2epid)
        vector_task_data = {epid: dict() for epid in it2epid if epid in self.all_episode_ids}
        if len(vector_task_data) > 0:
            for source in self.vector_task_sources:  # these are observations for next step!
                datum_id = self.source_to_str(source, is_vector_task=True)
                method, kwargs = source
                res = getattr(vector_task, method)(**kwargs)
                assert len(res) == len(it2epid)
                for datum, epid in zip(res, it2epid):
                    if epid in vector_task_data:
                        assert datum_id not in vector_task_data[epid]
                        vector_task_data[epid][datum_id] = datum

        # 3. gather rollout_sources (in phase with last_it2epid that stay alive)
        alive_set = set(alive)
        assert len(alive_set) == len(alive)
        alive_it2epid = [epid for it, epid in enumerate(self.last_it2epid) if it in alive_set]
        assert len(alive_it2epid) == len(it2epid) or len(self.last_it2epid) == 0
        rollout_data = {epid: dict() for epid in alive_it2epid if epid in self.all_episode_ids}
        if len(rollout_data) > 0 and rollout is not None:
            for source in self.rollout_sources:
                datum_id = self.source_to_str(source, is_vector_task=False)
                storage, path, episode_dim = source
                # Access storage
                res = getattr(rollout, storage)
                # Access sub-storage if path not empty
                if len(path) > 0:
                    res = res[rollout.reverse_flattened_spaces[tuple(path)]]
                # Select latest step
                res = res.narrow(
                    dim=0,  # step dimension
                    start=rollout.step - 1 if rollout.step > 0 else rollout.num_steps - 1,
                    length=1
                )
                for it, epid in enumerate(alive_it2epid):
                    if epid in rollout_data:
                        # Select current episode
                        datum = res.narrow(dim=episode_dim, start=it, length=1).to("cpu").detach().numpy()
                        assert datum_id not in rollout_data[epid]
                        rollout_data[epid][datum_id] = datum

        # 4. gather actor_critic_sources (in phase with last_it2epid)
        actor_critic_data = {epid: dict() for epid in self.last_it2epid if epid in self.all_episode_ids}
        if len(actor_critic_data) > 0 and actor_critic is not None:
            if self.actor_critic_source:
                probs = actor_critic.distributions.probs
                values = actor_critic.values
                for it, epid in enumerate(self.last_it2epid):
                    if epid in actor_critic_data:
                        # Select current episode (sampler axis will be reused as step axis)
                        prob = probs.narrow(dim=0, start=it, length=1).to("cpu").detach().numpy()
                        assert "actor_probs" not in actor_critic_data[epid]
                        actor_critic_data[epid]["actor_probs"] = prob
                        val = values.narrow(dim=0, start=it, length=1).to("cpu").detach().numpy()
                        assert "critic_value" not in actor_critic_data[epid]
                        actor_critic_data[epid]["critic_value"] = val

        # 5. append collected data to corresponding episodes
        for epid in vector_task_data:
            if epid in self.data:
                LOGGER.debug("Appending {} to {} with {}".format(list(vector_task_data[epid].keys()), epid, len(self.data[epid])))
                self.data[epid].append(vector_task_data[epid])
            else:
                LOGGER.debug("Starting appending {} to {}".format(list(vector_task_data[epid].keys()), epid))
                self.data[epid] = [vector_task_data[epid]]

        # e.g. update previous steps by storing tensors after pre-stored observation
        for epid in rollout_data:
            assert epid in self.data
            LOGGER.debug("Rollout updating {} to {} with {}".format(list(rollout_data[epid].keys()), epid, len(self.data[epid])))
            self.data[epid][-1].update(rollout_data[epid])

        for epid in actor_critic_data:
            assert epid in self.data
            LOGGER.debug("Actor-critic updating {} to {} with {}".format(list(actor_critic_data[epid].keys()), epid, len(self.data[epid])))
            self.data[epid][-1].update(actor_critic_data[epid])

        self.last_it2epid = it2epid

    # to be called by engine
    def read_and_reset(self):
        res, self.data = self.data, {}
        LOGGER.debug("Returning episodes {}".format(list(res.keys())))
        return res

    # to be called by logger
    def log(
            self,
            log_writer: SummaryWriter,
            task_outputs: Optional[List[Any]],
            render: Optional[List[Any]],
            num_steps: int
    ):
        for v in self.viz:
            LOGGER.debug("Logging {}".format(v.label))
            v.log(log_writer, task_outputs, render, num_steps)
