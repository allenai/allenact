import abc
import json
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
    # When debugging we don't want to use the interactive version of matplotlib
    # as it causes all sorts of problems.
    import pydevd
    import matplotlib

    matplotlib.use("agg")
except ImportError as _:
    pass

from matplotlib import pyplot as plt, markers
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
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
        start_marker_shape: str = "$\spadesuit$",
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
            lc = LineCollection(
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
                    "{}/{}_group{}".format(self.mode, self.label, page),
                    vid,
                    global_step=num_steps,
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
    def make_fig(self, episode_src: Sequence[np.ndarray], episode_id: str) -> Figure:
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
        rollout_source: Union[str, Sequence[str]] = ("memory", "rnn"),
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
            self._collect_actor_critic(actor_critic)

        if alive is not None and rollout is not None:
            # in phase with last_it2epid that stay alive
            self._collect_rollout(rollout, alive)

        # Always call this one last!
        if vector_task is not None:
            # in phase with identifiers of current episodes from vector_task
            self.last_it2epid = self._collect_vector_task(vector_task)

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
            # get_logger().debug("Logging {}".format(v.label))
            v.log(log_writer, task_outputs, render, num_steps)
