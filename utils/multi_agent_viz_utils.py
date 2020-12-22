from typing import Sequence, Any

import numpy as np
from matplotlib import pyplot as plt, markers
from matplotlib.collections import LineCollection

from utils.viz_utils import TrajectoryViz


class MultiTrajectoryViz(TrajectoryViz):
    def __init__(
        self,
        path_to_trajectory_prefix: Sequence[str] = ("task_info", "followed_path"),
        agent_suffixes: Sequence[str] = ("1", "2"),
        label: str = "trajectories",
        trajectory_plt_colormaps: Sequence[str] = ("cool", "spring"),
        marker_plt_colors: Sequence[Any] = ("blue", "orange"),
        axes_equal: bool = True,
        **other_base_kwargs,
    ):
        super().__init__(label=label, **other_base_kwargs)

        self.path_to_trajectory_prefix = list(path_to_trajectory_prefix)
        self.agent_suffixes = list(agent_suffixes)
        self.trajectory_plt_colormaps = list(trajectory_plt_colormaps)
        self.marker_plt_colors = marker_plt_colors
        self.axes_equal = axes_equal

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

        fig, ax = plt.subplots(figsize=self.figsize)
        for agent, cmap, marker_color in zip(
            self.agent_suffixes, self.trajectory_plt_colormaps, self.marker_plt_colors
        ):
            path = self.path_to_trajectory_prefix[:]
            path[-1] = path[-1] + agent
            trajectory = self._access(episode, path)

            x, y = [], []
            for xy in trajectory:
                x.append(float(self._access(xy, self.x)))
                y.append(float(self._access(xy, self.y)))

            colorline(x, y, zorder=1, cmap=cmap)

            start_marker = markers.MarkerStyle(marker=self.start_marker_shape)
            if self.path_to_rot_degrees is not None:
                rot_degrees = float(
                    self._access(trajectory[0], self.path_to_rot_degrees)
                )
                if self.adapt_rotation is not None:
                    rot_degrees = self.adapt_rotation(rot_degrees)
                start_marker._transform = start_marker.get_transform().rotate_deg(
                    rot_degrees
                )

            ax.scatter(
                [x[0]],
                [y[0]],
                marker=start_marker,
                zorder=2,
                s=self.start_marker_scale,
                color=marker_color,
            )
            ax.scatter(
                [x[-1]], [y[-1]], marker="s", color=marker_color
            )  # stop (square)

        if self.axes_equal:
            ax.set_aspect("equal", "box")
        ax.set_title(episode_id, fontsize=self.fontsize)
        ax.tick_params(axis="x", labelsize=self.fontsize)
        ax.tick_params(axis="y", labelsize=self.fontsize)

        return fig
