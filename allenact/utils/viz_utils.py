import abc
import json
import os
import sys
from collections import defaultdict
from typing import (
    Dict,
    Any,
    Union,
    Optional,
    List,
    Tuple,
    Sequence,
    Callable,
    cast,
    Set,
)

import numpy as np

from allenact.utils.experiment_utils import Builder
from allenact.utils.tensor_utils import SummaryWriter, tile_images, process_video

try:
    # Tensorflow not installed for testing
    from tensorflow.core.util import event_pb2
    from tensorflow.python.lib.io import tf_record

    _TF_AVAILABLE = True
except ImportError as _:
    event_pb2 = None
    tf_record = None

    _TF_AVAILABLE = False

import matplotlib

try:
    # When debugging we don't want to use the interactive version of matplotlib
    # as it causes all sorts of problems.

    # noinspection PyPackageRequirements
    import pydevd

    matplotlib.use("agg")
except ImportError as _:
    pass

import matplotlib.pyplot as plt
import matplotlib.markers as markers
import cv2

from allenact.utils.system import get_logger


class AbstractViz:
    def __init__(
        self,
        label: Optional[str] = None,
        vector_task_sources: Sequence[Tuple[str, Dict[str, Any]]] = (),
        rollout_sources: Sequence[Union[str, Sequence[str]]] = (),
        actor_critic_source: bool = False,
        **kwargs,  # accepts `max_episodes_in_group`
    ):
        self.label = label
        self.vector_task_sources = list(vector_task_sources)
        self.rollout_sources = [
            [entry] if isinstance(entry, str) else list(entry)
            for entry in rollout_sources
        ]
        self.actor_critic_source = actor_critic_source

        self.mode: Optional[str] = None
        self.path_to_id: Optional[Sequence[str]] = None
        self.episode_ids: Optional[List[Sequence[str]]] = None

        if "max_episodes_in_group" in kwargs:
            self.max_episodes_in_group = kwargs["max_episodes_in_group"]
            self.assigned_max_eps_in_group = True
        else:
            self.max_episodes_in_group = 8
            self.assigned_max_eps_in_group = False

    @staticmethod
    def _source_to_str(source, is_vector_task):
        source_type = "vector_task" if is_vector_task else "rollout_or_actor_critic"
        return "{}__{}".format(
            source_type,
            "__{}_sep__".format(source_type).join(["{}".format(s) for s in source]),
        )

    @staticmethod
    def _access(dictionary, path):
        path = path[::-1]
        while len(path) > 0:
            dictionary = dictionary[path.pop()]
        return dictionary

    def _auto_viz_order(self, task_outputs):
        if task_outputs is None:
            return None, None

        all_episodes = {
            self._access(episode, self.path_to_id): episode for episode in task_outputs
        }

        if self.episode_ids is None:
            all_episode_keys = list(all_episodes.keys())
            viz_order = []
            for page_start in range(
                0, len(all_episode_keys), self.max_episodes_in_group
            ):
                viz_order.append(
                    all_episode_keys[
                        page_start : page_start + self.max_episodes_in_group
                    ]
                )
            get_logger().debug("visualizing with order {}".format(viz_order))
        else:
            viz_order = self.episode_ids

        return viz_order, all_episodes

    def _setup(
        self,
        mode: str,
        path_to_id: Sequence[str],
        episode_ids: Optional[Sequence[Union[Sequence[str], str]]],
        max_episodes_in_group: int,
        force: bool = False,
    ):
        self.mode = mode
        self.path_to_id = list(path_to_id)
        if (self.episode_ids is None or force) and episode_ids is not None:
            self.episode_ids = (
                list(episode_ids)
                if not isinstance(episode_ids[0], str)
                else [list(cast(List[str], episode_ids))]
            )
        if not self.assigned_max_eps_in_group or force:
            self.max_episodes_in_group = max_episodes_in_group

    @abc.abstractmethod
    def log(
        self,
        log_writer: SummaryWriter,
        task_outputs: Optional[List[Any]],
        render: Optional[Dict[str, List[Dict[str, Any]]]],
        num_steps: int,
    ):
        raise NotImplementedError()


class TrajectoryViz(AbstractViz):
    def __init__(
        self,
        path_to_trajectory: Sequence[str] = ("task_info", "followed_path"),
        path_to_target_location: Optional[Sequence[str]] = (
            "task_info",
            "target_position",
        ),
        path_to_x: Sequence[str] = ("x",),
        path_to_y: Sequence[str] = ("z",),
        path_to_rot_degrees: Optional[Sequence[str]] = ("rotation", "y"),
        adapt_rotation: Optional[Callable[[float], float]] = None,
        label: str = "trajectory",
        figsize: Tuple[float, float] = (2, 2),
        fontsize: float = 5,
        start_marker_shape: str = r"$\spadesuit$",
        start_marker_scale: int = 100,
        **other_base_kwargs,
    ):
        super().__init__(label, **other_base_kwargs)
        self.path_to_trajectory = list(path_to_trajectory)
        self.path_to_target_location = (
            list(path_to_target_location)
            if path_to_target_location is not None
            else None
        )
        self.adapt_rotation = adapt_rotation
        self.x = list(path_to_x)
        self.y = list(path_to_y)
        self.path_to_rot_degrees = (
            list(path_to_rot_degrees) if path_to_rot_degrees is not None else None
        )
        self.figsize = figsize
        self.fontsize = fontsize
        self.start_marker_shape = start_marker_shape
        self.start_marker_scale = start_marker_scale

    def log(
        self,
        log_writer: SummaryWriter,
        task_outputs: Optional[List[Any]],
        render: Optional[Dict[str, List[Dict[str, Any]]]],
        num_steps: int,
    ):
        viz_order, all_episodes = self._auto_viz_order(task_outputs)
        if viz_order is None:
            get_logger().debug("trajectory viz returning without visualizing")
            return

        for page, current_ids in enumerate(viz_order):
            figs = []
            for episode_id in current_ids:
                # assert episode_id in all_episodes
                if episode_id not in all_episodes:
                    get_logger().warning(
                        "skipping viz for missing episode {}".format(episode_id)
                    )
                    continue
                figs.append(self.make_fig(all_episodes[episode_id], episode_id))
            if len(figs) == 0:
                continue
            log_writer.add_figure(
                "{}/{}_group{}".format(self.mode, self.label, page),
                figs,
                global_step=num_steps,
            )
            plt.close(
                "all"
            )  # close all current figures (SummaryWriter already closes all figures we log)

    def make_fig(self, episode, episode_id):
        # From https://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
        def colorline(
            x,
            y,
            z=None,
            cmap=plt.get_cmap("cool"),
            norm=plt.Normalize(0.0, 1.0),
            linewidth=2,
            alpha=1.0,
            zorder=1,
        ):
            """Plot a colored line with coordinates x and y.

            Optionally specify colors in the array z

            Optionally specify a colormap, a norm function and a line width.
            """

            def make_segments(x, y):
                """Create list of line segments from x and y coordinates, in
                the correct format for LineCollection:

                an array of the form  numlines x (points per line) x 2
                (x and y) array
                """
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                return segments

            # Default colors equally spaced on [0,1]:
            if z is None:
                z = np.linspace(0.0, 1.0, len(x))

            # Special case if a single number:
            if not hasattr(
                z, "__iter__"
            ):  # to check for numerical input -- this is a hack
                z = np.array([z])

            z = np.asarray(z)

            segments = make_segments(x, y)
            lc = matplotlib.collections.LineCollection(
                segments,
                array=z,
                cmap=cmap,
                norm=norm,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )

            ax = plt.gca()
            ax.add_collection(lc)

            return lc

        trajectory = self._access(episode, self.path_to_trajectory)

        x, y = [], []
        for xy in trajectory:
            x.append(float(self._access(xy, self.x)))
            y.append(float(self._access(xy, self.y)))

        fig, ax = plt.subplots(figsize=self.figsize)
        colorline(x, y, zorder=1)

        start_marker = markers.MarkerStyle(marker=self.start_marker_shape)
        if self.path_to_rot_degrees is not None:
            rot_degrees = float(self._access(trajectory[0], self.path_to_rot_degrees))
            if self.adapt_rotation is not None:
                rot_degrees = self.adapt_rotation(rot_degrees)
            start_marker._transform = start_marker.get_transform().rotate_deg(
                rot_degrees
            )

        ax.scatter(
            [x[0]], [y[0]], marker=start_marker, zorder=2, s=self.start_marker_scale
        )
        ax.scatter([x[-1]], [y[-1]], marker="s")  # stop

        if self.path_to_target_location is not None:
            target = self._access(episode, self.path_to_target_location)
            ax.scatter(
                [float(self._access(target, self.x))],
                [float(self._access(target, self.y))],
                marker="*",
            )

        ax.set_title(episode_id, fontsize=self.fontsize)
        ax.tick_params(axis="x", labelsize=self.fontsize)
        ax.tick_params(axis="y", labelsize=self.fontsize)

        return fig


class AgentViewViz(AbstractViz):
    def __init__(
        self,
        label: str = "agent_view",
        max_clip_length: int = 100,  # control memory used when converting groups of images into clips
        max_video_length: int = -1,  # no limit, if > 0, limit the maximum video length (discard last frames)
        vector_task_source: Tuple[str, Dict[str, Any]] = (
            "render",
            {"mode": "raw_rgb_list"},
        ),
        episode_ids: Optional[Sequence[Union[Sequence[str], str]]] = None,
        fps: int = 4,
        max_render_size: int = 400,
        **other_base_kwargs,
    ):
        super().__init__(
            label, vector_task_sources=[vector_task_source], **other_base_kwargs,
        )
        self.max_clip_length = max_clip_length
        self.max_video_length = max_video_length
        self.fps = fps
        self.max_render_size = max_render_size

        self.episode_ids = (
            (
                list(episode_ids)
                if not isinstance(episode_ids[0], str)
                else [list(cast(List[str], episode_ids))]
            )
            if episode_ids is not None
            else None
        )

    def log(
        self,
        log_writer: SummaryWriter,
        task_outputs: Optional[List[Any]],
        render: Optional[Dict[str, List[Dict[str, Any]]]],
        num_steps: int,
    ):
        if render is None:
            return

        datum_id = self._source_to_str(self.vector_task_sources[0], is_vector_task=True)

        viz_order, _ = self._auto_viz_order(task_outputs)
        if viz_order is None:
            get_logger().debug("agent view viz returning without visualizing")
            return

        for page, current_ids in enumerate(viz_order):
            images = []  # list of lists of rgb frames
            for episode_id in current_ids:
                # assert episode_id in render
                if episode_id not in render:
                    get_logger().warning(
                        "skipping viz for missing episode {}".format(episode_id)
                    )
                    continue
                images.append(
                    [
                        self._overlay_label(step[datum_id], episode_id)
                        for step in render[episode_id]
                    ]
                )
            if len(images) == 0:
                continue
            vid = self.make_vid(images)
            if vid is not None:
                log_writer.add_vid(
                    f"{self.mode}/{self.label}_group{page}", vid, global_step=num_steps,
                )

    @staticmethod
    def _overlay_label(
        img,
        text,
        pos=(0, 0),
        bg_color=(255, 255, 255),
        fg_color=(0, 0, 0),
        scale=0.4,
        thickness=1,
        margin=2,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
    ):
        txt_size = cv2.getTextSize(text, font_face, scale, thickness)

        end_x = pos[0] + txt_size[0][0] + margin
        end_y = pos[1]

        pos = (pos[0], pos[1] + txt_size[0][1] + margin)

        cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
        cv2.putText(
            img=img,
            text=text,
            org=pos,
            fontFace=font_face,
            fontScale=scale,
            color=fg_color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        return img

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

        return process_video(
            frames, self.max_clip_length, self.max_video_length, fps=self.fps
        )


class AbstractTensorViz(AbstractViz):
    def __init__(
        self,
        rollout_source: Union[str, Sequence[str]],
        label: Optional[str] = None,
        figsize: Tuple[float, float] = (3, 3),
        **other_base_kwargs,
    ):
        if label is None:
            if isinstance(rollout_source, str):
                label = rollout_source[:]
            else:
                label = "/".join(rollout_source)

        super().__init__(label, rollout_sources=[rollout_source], **other_base_kwargs)

        self.figsize = figsize
        self.datum_id = self._source_to_str(
            self.rollout_sources[0], is_vector_task=False
        )

    def log(
        self,
        log_writer: SummaryWriter,
        task_outputs: Optional[List[Any]],
        render: Optional[Dict[str, List[Dict[str, Any]]]],
        num_steps: int,
    ):
        if render is None:
            return

        viz_order, _ = self._auto_viz_order(task_outputs)
        if viz_order is None:
            get_logger().debug("tensor viz returning without visualizing")
            return

        for page, current_ids in enumerate(viz_order):
            figs = []
            for episode_id in current_ids:
                if episode_id not in render or len(render[episode_id]) == 0:
                    get_logger().warning(
                        "skipping viz for missing or 0-length episode {}".format(
                            episode_id
                        )
                    )
                    continue
                episode_src = [
                    step[self.datum_id]
                    for step in render[episode_id]
                    if self.datum_id in step
                ]
                if len(episode_src) > 0:
                    # If the last episode for an inference worker is of length 1, there's no captured rollout sources
                    figs.append(self.make_fig(episode_src, episode_id))
            if len(figs) == 0:
                continue
            log_writer.add_figure(
                "{}/{}_group{}".format(self.mode, self.label, page),
                figs,
                global_step=num_steps,
            )
            plt.close(
                "all"
            )  # close all current figures (SummaryWriter already closes all figures we log)

    @abc.abstractmethod
    def make_fig(
        self, episode_src: Sequence[np.ndarray], episode_id: str
    ) -> matplotlib.figure.Figure:
        raise NotImplementedError()


class TensorViz1D(AbstractTensorViz):
    def __init__(
        self,
        rollout_source: Union[str, Sequence[str]] = "action_log_probs",
        label: Optional[str] = None,
        figsize: Tuple[float, float] = (3, 3),
        **other_base_kwargs,
    ):
        super().__init__(rollout_source, label, figsize, **other_base_kwargs)

    def make_fig(self, episode_src, episode_id):
        assert episode_src[0].size == 1

        # Concatenate along step axis (0)
        seq = np.concatenate(episode_src, axis=0).squeeze()  # remove all singleton dims

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(seq)
        ax.set_title(episode_id)

        ax.set_aspect("auto")
        plt.tight_layout()

        return fig


class TensorViz2D(AbstractTensorViz):
    def __init__(
        self,
        rollout_source: Union[str, Sequence[str]] = ("memory_first_last", "rnn"),
        label: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 10),
        fontsize: float = 5,
        **other_base_kwargs,
    ):
        super().__init__(rollout_source, label, figsize, **other_base_kwargs)
        self.fontsize = fontsize

    def make_fig(self, episode_src, episode_id):
        # Concatenate along step axis (0)
        seq = np.concatenate(
            episode_src, axis=0
        ).squeeze()  # remove num_layers if it's equal to 1, else die
        assert len(seq.shape) == 2, "No support for higher-dimensions"

        # get_logger().debug("basic {} h render {}".format(episode_id, seq[:10, 0]))

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.matshow(seq)

        ax.set_xlabel(episode_id, fontsize=self.fontsize)
        ax.tick_params(axis="x", labelsize=self.fontsize)
        ax.tick_params(axis="y", labelsize=self.fontsize)
        ax.tick_params(bottom=False)

        ax.set_aspect("auto")
        plt.tight_layout()

        return fig


class ActorViz(AbstractViz):
    def __init__(
        self,
        label: str = "action_probs",
        action_names_path: Optional[Sequence[str]] = ("task_info", "action_names"),
        figsize: Tuple[float, float] = (1, 5),
        fontsize: float = 5,
        **other_base_kwargs,
    ):
        super().__init__(label, actor_critic_source=True, **other_base_kwargs)
        self.action_names_path: Optional[Sequence[str]] = (
            list(action_names_path) if action_names_path is not None else None
        )
        self.figsize = figsize
        self.fontsize = fontsize
        self.action_names: Optional[List[str]] = None

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
            self.action_names = list(
                self._access(task_outputs[0], self.action_names_path)
            )

        viz_order, _ = self._auto_viz_order(task_outputs)
        if viz_order is None:
            get_logger().debug("actor viz returning without visualizing")
            return

        for page, current_ids in enumerate(viz_order):
            figs = []
            for episode_id in current_ids:
                # assert episode_id in render
                if episode_id not in render:
                    get_logger().warning(
                        "skipping viz for missing episode {}".format(episode_id)
                    )
                    continue
                episode_src = [
                    step["actor_probs"]
                    for step in render[episode_id]
                    if "actor_probs" in step
                ]
                assert len(episode_src) == len(render[episode_id])
                figs.append(self.make_fig(episode_src, episode_id))
            if len(figs) == 0:
                continue
            log_writer.add_figure(
                "{}/{}_group{}".format(self.mode, self.label, page),
                figs,
                global_step=num_steps,
            )
            plt.close(
                "all"
            )  # close all current figures (SummaryWriter already closes all figures we log)

    def make_fig(self, episode_src, episode_id):
        # Concatenate along step axis (0, reused from kept sampler axis)
        mat = np.concatenate(episode_src, axis=0)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.matshow(mat)

        if self.action_names is not None:
            assert len(self.action_names) == mat.shape[-1]
            ax.set_xticklabels([""] + self.action_names, rotation="vertical")

        ax.set_xlabel(episode_id, fontsize=self.fontsize)
        ax.tick_params(axis="x", labelsize=self.fontsize)
        ax.tick_params(axis="y", labelsize=self.fontsize)
        ax.tick_params(bottom=False)

        # Gridlines based on minor ticks
        ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
        ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=0.05)
        ax.tick_params(
            axis="both", which="minor", left=False, top=False, right=False, bottom=False
        )

        ax.set_aspect("auto")
        plt.tight_layout()
        return fig


class VizSuite(AbstractViz):
    def __init__(
        self,
        episode_ids: Optional[Sequence[Union[Sequence[str], str]]] = None,
        path_to_id: Sequence[str] = ("task_info", "id"),
        mode: str = "valid",
        force_episodes_and_max_episodes_in_group: bool = False,
        max_episodes_in_group: int = 8,
        *viz,
        **kw_viz,
    ):
        super().__init__(max_episodes_in_group=max_episodes_in_group)
        self._setup(
            mode=mode,
            path_to_id=path_to_id,
            episode_ids=episode_ids,
            max_episodes_in_group=max_episodes_in_group,
        )
        self.force_episodes_and_max_episodes_in_group = (
            force_episodes_and_max_episodes_in_group
        )

        self.all_episode_ids = self._episodes_set()

        self.viz = [
            v() if isinstance(v, Builder) else v
            for v in viz
            if isinstance(v, Builder) or isinstance(v, AbstractViz)
        ] + [
            v() if isinstance(v, Builder) else v
            for k, v in kw_viz.items()
            if isinstance(v, Builder) or isinstance(v, AbstractViz)
        ]

        self.max_render_size: Optional[int] = None

        (
            self.rollout_sources,
            self.vector_task_sources,
            self.actor_critic_source,
        ) = self._setup_sources()

        self.data: Dict[
            str, List[Dict]
        ] = {}  # dict of episode id to list of dicts with collected data
        self.last_it2epid: List[str] = []

    def _setup_sources(self):
        rollout_sources, vector_task_sources = [], []
        labels = []
        actor_critic_source = False
        new_episodes = []
        for v in self.viz:
            labels.append(v.label)
            rollout_sources += v.rollout_sources
            vector_task_sources += v.vector_task_sources
            actor_critic_source |= v.actor_critic_source

            if (
                v.episode_ids is not None
                and not self.force_episodes_and_max_episodes_in_group
            ):
                cur_episodes = self._episodes_set(v.episode_ids)
                for ep in cur_episodes:
                    if (
                        self.all_episode_ids is not None
                        and ep not in self.all_episode_ids
                    ):
                        new_episodes.append(ep)
                        get_logger().info(
                            "Added new episode {} from {}".format(ep, v.label)
                        )

            v._setup(
                mode=self.mode,
                path_to_id=self.path_to_id,
                episode_ids=self.episode_ids,
                max_episodes_in_group=self.max_episodes_in_group,
                force=self.force_episodes_and_max_episodes_in_group,
            )

            if isinstance(v, AgentViewViz):
                self.max_render_size = v.max_render_size

        get_logger().info("Logging labels {}".format(labels))

        if len(new_episodes) > 0:
            get_logger().info("Added new episodes {}".format(new_episodes))
            self.episode_ids.append(new_episodes)  # new group with all added episodes
            self.all_episode_ids = self._episodes_set()

        rol_flat = {json.dumps(src, sort_keys=True): src for src in rollout_sources}
        vt_flat = {json.dumps(src, sort_keys=True): src for src in vector_task_sources}

        rol_keys = list(set(rol_flat.keys()))
        vt_keys = list(set(vt_flat.keys()))

        return (
            [rol_flat[k] for k in rol_keys],
            [vt_flat[k] for k in vt_keys],
            actor_critic_source,
        )

    def _episodes_set(self, episode_list=None) -> Optional[Set[str]]:
        source = self.episode_ids if episode_list is None else episode_list
        if source is None:
            return None

        all_episode_ids: List[str] = []
        for group in source:
            all_episode_ids += group
        return set(all_episode_ids)

    def empty(self):
        return len(self.data) == 0

    def _update(self, collected_data):
        for epid in collected_data:
            assert epid in self.data
            self.data[epid][-1].update(collected_data[epid])

    def _append(self, vector_task_data):
        for epid in vector_task_data:
            if epid in self.data:
                self.data[epid].append(vector_task_data[epid])
            else:
                self.data[epid] = [vector_task_data[epid]]

    def _collect_actor_critic(self, actor_critic):
        actor_critic_data = {
            epid: dict()
            for epid in self.last_it2epid
            if self.all_episode_ids is None or epid in self.all_episode_ids
        }
        if len(actor_critic_data) > 0 and actor_critic is not None:
            if self.actor_critic_source:
                # TODO this code only supports Discrete action spaces!
                probs = (
                    actor_critic.distributions.probs
                )  # step (=1) x sampler x agent (=1) x action
                values = actor_critic.values  # step x sampler x agent x 1
                for it, epid in enumerate(self.last_it2epid):
                    if epid in actor_critic_data:
                        # Select current episode (sampler axis will be reused as step axis)
                        prob = (
                            # probs.narrow(dim=0, start=it, length=1)  # works for sampler x action
                            probs.narrow(
                                dim=1, start=it, length=1
                            )  # step x sampler x agent x action -> step x 1 x agent x action
                            .squeeze(
                                0
                            )  # step x 1 x agent x action -> 1 x agent x action
                            # .squeeze(-2)  # 1 x agent x action -> 1 x action
                            .to("cpu")
                            .detach()
                            .numpy()
                        )
                        assert "actor_probs" not in actor_critic_data[epid]
                        actor_critic_data[epid]["actor_probs"] = prob
                        val = (
                            # values.narrow(dim=0, start=it, length=1)  # works for sampler x 1
                            values.narrow(
                                dim=1, start=it, length=1
                            )  # step x sampler x agent x 1 -> step x 1 x agent x 1
                            .squeeze(0)  # step x 1 x agent x 1 -> 1 x agent x 1
                            # .squeeze(-2)  # 1 x agent x 1 -> 1 x 1
                            .to("cpu")
                            .detach()
                            .numpy()
                        )
                        assert "critic_value" not in actor_critic_data[epid]
                        actor_critic_data[epid]["critic_value"] = val

        self._update(actor_critic_data)

    def _collect_rollout(self, rollout, alive):
        alive_set = set(alive)
        assert len(alive_set) == len(alive)
        alive_it2epid = [
            epid for it, epid in enumerate(self.last_it2epid) if it in alive_set
        ]
        rollout_data = {
            epid: dict()
            for epid in alive_it2epid
            if self.all_episode_ids is None or epid in self.all_episode_ids
        }
        if len(rollout_data) > 0 and rollout is not None:
            for source in self.rollout_sources:
                datum_id = self._source_to_str(source, is_vector_task=False)

                storage, path = source[0], source[1:]

                # Access storage
                res = getattr(rollout, storage)
                episode_dim = rollout.dim_names.index("sampler")

                # Access sub-storage if path not empty
                if len(path) > 0:
                    if storage == "memory_first_last":
                        storage = "memory"

                    flattened_name = rollout.unflattened_to_flattened[storage][
                        tuple(path)
                    ]
                    # for path_step in path:
                    #     res = res[path_step]
                    res = res[flattened_name]
                    res, episode_dim = res

                if rollout.step > 0:
                    if rollout.step > res.shape[0]:
                        # e.g. rnn with only latest memory saved
                        rollout_step = res.shape[0] - 1
                    else:
                        rollout_step = rollout.step - 1
                else:
                    if rollout.num_steps - 1 < res.shape[0]:
                        rollout_step = rollout.num_steps - 1
                    else:
                        # e.g. rnn with only latest memory saved
                        rollout_step = res.shape[0] - 1

                # Select latest step
                res = res.narrow(
                    dim=0, start=rollout_step, length=1,  # step dimension
                )  # 1 x ... x sampler x ...

                # get_logger().debug("basic collect h {}".format(res[..., 0]))

                for it, epid in enumerate(alive_it2epid):
                    if epid in rollout_data:
                        # Select current episode and remove episode/sampler axis
                        datum = (
                            res.narrow(dim=episode_dim, start=it, length=1)
                            .squeeze(axis=episode_dim)
                            .to("cpu")
                            .detach()
                            .numpy()
                        )  # 1 x ... (no sampler dim)
                        # get_logger().debug("basic collect ep {} h {}".format(epid, res[..., 0]))
                        assert datum_id not in rollout_data[epid]
                        rollout_data[epid][
                            datum_id
                        ] = datum.copy()  # copy needed when running on CPU!

        self._update(rollout_data)

    def _collect_vector_task(self, vector_task):
        it2epid = [
            self._access(info, self.path_to_id[1:])
            for info in vector_task.attr("task_info")
        ]
        # get_logger().debug("basic epids {}".format(it2epid))

        def limit_spatial_res(data: np.ndarray, max_size=400):
            if data.shape[0] <= max_size and data.shape[1] <= max_size:
                return data
            else:
                f = float(max_size) / max(data.shape[0], data.shape[1])
                size = (int(data.shape[1] * f), int(data.shape[0] * f))
                return cv2.resize(data, size, 0, 0, interpolation=cv2.INTER_AREA)

        vector_task_data = {
            epid: dict()
            for epid in it2epid
            if self.all_episode_ids is None or epid in self.all_episode_ids
        }
        if len(vector_task_data) > 0:
            for (
                source
            ) in self.vector_task_sources:  # these are observations for next step!
                datum_id = self._source_to_str(source, is_vector_task=True)
                method, kwargs = source
                res = getattr(vector_task, method)(**kwargs)
                if not isinstance(res, Sequence):
                    assert len(it2epid) == 1
                    res = [res]
                if method == "render":
                    res = [limit_spatial_res(r, self.max_render_size) for r in res]
                assert len(res) == len(it2epid)
                for datum, epid in zip(res, it2epid):
                    if epid in vector_task_data:
                        assert datum_id not in vector_task_data[epid]
                        vector_task_data[epid][datum_id] = datum

        self._append(vector_task_data)

        return it2epid

    # to be called by engine
    def collect(self, vector_task=None, alive=None, rollout=None, actor_critic=None):
        if actor_critic is not None:
            # in phase with last_it2epid
            try:
                self._collect_actor_critic(actor_critic)
            except (AssertionError, RuntimeError):
                get_logger().debug(
                    msg=f"Failed collect (actor_critic) for viz due to exception:",
                    exc_info=sys.exc_info(),
                )
                get_logger().error(f"Failed collect (actor_critic) for viz")

        if alive is not None and rollout is not None:
            # in phase with last_it2epid that stay alive
            try:
                self._collect_rollout(rollout=rollout, alive=alive)
            except (AssertionError, RuntimeError):
                get_logger().debug(
                    msg=f"Failed collect (rollout) for viz due to exception:",
                    exc_info=sys.exc_info(),
                )
                get_logger().error(f"Failed collect (rollout) for viz")

        # Always call this one last!
        if vector_task is not None:
            # in phase with identifiers of current episodes from vector_task
            try:
                self.last_it2epid = self._collect_vector_task(vector_task)
            except (AssertionError, RuntimeError):
                get_logger().debug(
                    msg=f"Failed collect (vector_task) for viz due to exception:",
                    exc_info=sys.exc_info(),
                )
                get_logger().error(f"Failed collect (vector_task) for viz")

    def read_and_reset(self) -> Dict[str, List[Dict[str, Any]]]:
        res = self.data
        self.data = {}
        # get_logger().debug("Returning episodes {}".format(list(res.keys())))
        return res

    # to be called by logger
    def log(
        self,
        log_writer: SummaryWriter,
        task_outputs: Optional[List[Any]],
        render: Optional[Dict[str, List[Dict[str, Any]]]],
        num_steps: int,
    ):
        for v in self.viz:
            try:
                v.log(log_writer, task_outputs, render, num_steps)
            except (AssertionError, RuntimeError):
                get_logger().debug(
                    msg=f"Dropped {v.label} viz due to exception:",
                    exc_info=sys.exc_info(),
                )
                get_logger().error(f"Dropped {v.label} viz")


class TensorboardSummarizer:
    """Assumption: tensorboard tags/labels include a valid/test/train substr indicating the data modality"""

    def __init__(
        self,
        experiment_to_train_events_paths_map: Dict[str, Sequence[str]],
        experiment_to_test_events_paths_map: Dict[str, Sequence[str]],
        eval_min_mega_steps: Optional[Sequence[float]] = None,
        tensorboard_tags_to_labels_map: Optional[Dict[str, str]] = None,
        tensorboard_output_summary_folder: str = "tensorboard_plotter_output",
    ):
        if not _TF_AVAILABLE:
            raise ImportError(
                "Please install tensorflow e.g. with `pip install tensorflow` to enable TensorboardSummarizer"
            )

        self.experiment_to_train_events_paths_map = experiment_to_train_events_paths_map
        self.experiment_to_test_events_paths_map = experiment_to_test_events_paths_map
        train_experiments = set(list(experiment_to_train_events_paths_map.keys()))
        test_experiments = set(list(experiment_to_test_events_paths_map.keys()))
        assert (train_experiments - test_experiments) in [set(), train_experiments,], (
            f"`experiment_to_test_events_paths_map` must have identical keys (experiment names) to those"
            f" in `experiment_to_train_events_paths_map`, or be empty."
            f" Got {train_experiments} train keys and {test_experiments} test keys."
        )

        self.eval_min_mega_steps = eval_min_mega_steps
        self.tensorboard_tags_to_labels_map = tensorboard_tags_to_labels_map
        if self.tensorboard_tags_to_labels_map is not None:
            for tag, label in self.tensorboard_tags_to_labels_map.items():
                assert ("valid" in label) + ("train" in label) + (
                    "test" in label
                ) == 1, (
                    f"One (and only one) of {'train', 'valid', 'test'} must be part of the label for"
                    f" tag {tag} ({label} given)."
                )
        self.tensorboard_output_summary_folder = tensorboard_output_summary_folder

        self.train_data = self._read_tensorflow_experiment_events(
            self.experiment_to_train_events_paths_map
        )
        self.test_data = self._read_tensorflow_experiment_events(
            self.experiment_to_test_events_paths_map
        )

    def _read_tensorflow_experiment_events(
        self, experiment_to_events_paths_map, skip_map=False
    ):
        def my_summary_iterator(path):
            try:
                for r in tf_record.tf_record_iterator(path):
                    yield event_pb2.Event.FromString(r)
            except IOError:
                get_logger().debug(f"IOError for path {path}")
                return None

        collected_data = {}
        for experiment_name, path_list in experiment_to_events_paths_map.items():
            experiment_data = defaultdict(list)
            for filename_path in path_list:
                for event in my_summary_iterator(filename_path):
                    if event is None:
                        break
                    for value in event.summary.value:
                        if self.tensorboard_tags_to_labels_map is None or skip_map:
                            label = value.tag
                        elif value.tag in self.tensorboard_tags_to_labels_map:
                            label = self.tensorboard_tags_to_labels_map[value.tag]
                        else:
                            continue
                        experiment_data[label].append(
                            dict(
                                score=value.simple_value,
                                time=event.wall_time,
                                steps=event.step,
                            )
                        )
            collected_data[experiment_name] = experiment_data

        return collected_data

    def _eval_vs_train_time_steps(self, eval_data, train_data):
        min_mega_steps = self.eval_min_mega_steps
        if min_mega_steps is None:
            min_mega_steps = [(item["steps"] - 1) / 1e6 for item in eval_data]

        scores, times, steps = [], [], []

        i, t, last_i = 0, 0, -1
        while len(times) < len(min_mega_steps):
            while eval_data[i]["steps"] / min_mega_steps[len(times)] / 1e6 < 1:
                i += 1
            while train_data[t]["steps"] / min_mega_steps[len(times)] / 1e6 < 1:
                t += 1

            # step might be missing in valid! (and would duplicate future value at previous steps!)
            # solution: move forward last entry's time if no change in i (instead of new entry)
            if i == last_i:
                times[-1] = train_data[t]["time"]
            else:
                scores.append(eval_data[i]["score"])
                times.append(train_data[t]["time"])
                steps.append(eval_data[i]["steps"])

            last_i = i

        scores.insert(0, train_data[0]["score"])
        times.insert(0, train_data[0]["time"])
        steps.insert(0, 0)

        return scores, times, steps

    def _train_vs_time_steps(self, train_data):
        last_eval_step = (
            self.eval_min_mega_steps[-1] * 1e6
            if self.eval_min_mega_steps is not None
            else float("inf")
        )

        scores = [train_data[0]["score"]]
        times = [train_data[0]["time"]]
        steps = [train_data[0]["steps"]]

        t = 1
        while steps[-1] < last_eval_step and t < len(train_data):
            scores.append(train_data[t]["score"])
            times.append(train_data[t]["time"])
            steps.append(train_data[t]["steps"])
            t += 1

        return scores, times, steps

    def make_tensorboard_summary(self):
        all_experiments = list(self.experiment_to_train_events_paths_map.keys())

        for experiment_name in all_experiments:
            summary_writer = SummaryWriter(
                os.path.join(self.tensorboard_output_summary_folder, experiment_name)
            )

            test_labels = (
                sorted(list(self.test_data[experiment_name].keys()))
                if len(self.test_data) > 0
                else []
            )
            for test_label in test_labels:
                train_label = test_label.replace("valid", "test").replace(
                    "test", "train"
                )
                if train_label not in self.train_data[experiment_name]:
                    print(
                        f"Missing matching 'train' label {train_label} for eval label {test_label}. Skipping"
                    )
                    continue
                train_data = self.train_data[experiment_name][train_label]
                test_data = self.test_data[experiment_name][test_label]
                scores, times, steps = self._eval_vs_train_time_steps(
                    test_data, train_data
                )
                for score, t, step in zip(scores, times, steps):
                    summary_writer.add_scalar(
                        test_label, score, global_step=step, walltime=t
                    )

            valid_labels = sorted(
                [
                    key
                    for key in list(self.train_data[experiment_name].keys())
                    if "valid" in key
                ]
            )
            for valid_label in valid_labels:
                train_label = valid_label.replace("valid", "train")
                assert (
                    train_label in self.train_data[experiment_name]
                ), f"Missing matching 'train' label {train_label} for valid label {valid_label}"
                train_data = self.train_data[experiment_name][train_label]
                valid_data = self.train_data[experiment_name][valid_label]
                scores, times, steps = self._eval_vs_train_time_steps(
                    valid_data, train_data
                )
                for score, t, step in zip(scores, times, steps):
                    summary_writer.add_scalar(
                        valid_label, score, global_step=step, walltime=t
                    )

            train_labels = sorted(
                [
                    key
                    for key in list(self.train_data[experiment_name].keys())
                    if "train" in key
                ]
            )
            for train_label in train_labels:
                scores, times, steps = self._train_vs_time_steps(
                    self.train_data[experiment_name][train_label]
                )
                for score, t, step in zip(scores, times, steps):
                    summary_writer.add_scalar(
                        train_label, score, global_step=step, walltime=t
                    )

            summary_writer.close()
