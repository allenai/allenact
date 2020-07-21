import copy
import math
from typing import Tuple, Sequence, Union, Dict, Optional, Any
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image, ImageDraw
from ai2thor.controller import Controller

from utils.viz_utils import TrajectoryViz
from utils.system import get_logger


class ThorPositionTo2DFrameTranslator(object):
    def __init__(self, frame_shape_rows_cols, cam_position, orth_size):
        self.frame_shape = frame_shape_rows_cols
        self.lower_left = np.array((cam_position[0], cam_position[2])) - orth_size
        self.span = 2 * orth_size

    def __call__(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position

        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )


def get_agent_map_data(env, viz_resolution):
    env.step({"action": "ToggleMapView"})
    cam_position = env.last_event.metadata["cameraPosition"]
    cam_orth_size = env.last_event.metadata["cameraOrthSize"]
    pos_translator = ThorPositionTo2DFrameTranslator(
        viz_resolution, position_to_tuple(cam_position), cam_orth_size
    )
    to_return = {
        "frame": env.last_event.frame,
        "cam_position": cam_position,
        "cam_orth_size": cam_orth_size,
        "pos_translator": pos_translator,
    }
    env.step({"action": "ToggleMapView"})
    return to_return


def position_to_tuple(position):
    return position["x"], position["y"], position["z"]


def add_lines_to_map(ps, frame, pos_translator, opacity, color=None):
    if len(ps) <= 1:
        return frame
    if color is None:
        color = (255, 0, 0)

    img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
    img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA

    opacity = int(round(255 * opacity))  # Define transparency for the triangle.
    draw = ImageDraw.Draw(img2)
    for i in range(len(ps) - 1):
        draw.line(
            tuple(reversed(pos_translator(ps[i])))
            + tuple(reversed(pos_translator(ps[i + 1]))),
            fill=color + (opacity,),
            width=int(frame.shape[0] / 100),
        )

    img = Image.alpha_composite(img1, img2)
    return np.array(img.convert("RGB"))


def add_line_to_map(p0, p1, frame, pos_translator, opacity, color=None):
    if p0 == p1:
        return frame
    if color is None:
        color = (255, 0, 0)

    img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
    img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA

    opacity = int(round(255 * opacity))  # Define transparency for the triangle.
    draw = ImageDraw.Draw(img2)
    draw.line(
        tuple(reversed(pos_translator(p0))) + tuple(reversed(pos_translator(p1))),
        fill=color + (opacity,),
        width=int(frame.shape[0] / 100),
    )

    img = Image.alpha_composite(img1, img2)
    return np.array(img.convert("RGB"))


def add_agent_view_triangle(
    position, rotation, frame, pos_translator, scale=1.0, opacity=0.1
):
    p0 = np.array((position[0], position[2]))
    p1 = copy.copy(p0)
    p2 = copy.copy(p0)

    theta = -2 * math.pi * (rotation["y"] / 360.0)
    rotation_mat = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    offset1 = scale * np.array([-1 / 2.0, 1])
    offset2 = scale * np.array([1 / 2.0, 1])

    p1 += np.matmul(rotation_mat, offset1)
    p2 += np.matmul(rotation_mat, offset2)

    img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
    img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA

    opacity = int(round(255 * opacity))  # Define transparency for the triangle.
    points = [tuple(reversed(pos_translator(p))) for p in [p0, p1, p2]]
    draw = ImageDraw.Draw(img2)
    draw.polygon(points, fill=(255, 255, 255, opacity))

    img = Image.alpha_composite(img1, img2)
    return np.array(img.convert("RGB"))


def visualize_agent_path(
    positions,
    frame,
    pos_translator,
    single_color=False,
    view_triangle_only_on_last=False,
    disable_view_triangle=False,
    line_opacity=1.0,
):

    if single_color:
        frame = add_lines_to_map(
            list(map(position_to_tuple, positions)),
            frame,
            pos_translator,
            line_opacity,
            (0, 255, 0),
        )
    else:
        import colour as col

        colors = list(col.Color("red").range_to(col.Color("green"), len(positions) - 1))
        for i in range(len(positions) - 1):
            frame = add_line_to_map(
                position_to_tuple(positions[i]),
                position_to_tuple(positions[i + 1]),
                frame,
                pos_translator,
                opacity=line_opacity,
                color=tuple(map(lambda x: int(round(255 * x)), colors[i].rgb)),
            )

    if view_triangle_only_on_last:
        positions = [positions[-1]]
    if disable_view_triangle:
        positions = []
    for position in positions:
        frame = add_agent_view_triangle(
            position_to_tuple(position),
            rotation=position["rotation"],
            frame=frame,
            pos_translator=pos_translator,
            opacity=0.05 + view_triangle_only_on_last * 0.2,
        )
    return frame


def dump_top_down_view(c, room_name, folder_name):
    get_logger().debug(
        "Dumping {}".format(os.path.join(folder_name, "{}.png".format(room_name)))
    )

    os.makedirs(folder_name, exist_ok=True)

    c.reset(room_name)
    c.step({"action": "Initialize", "gridSize": 0.1, "makeAgentsVisible": False})
    c.step({"action": "ToggleMapView"})
    top_down_view = c.last_event.cv2img

    cv2.imwrite(os.path.join(folder_name, "{}.png".format(room_name)), top_down_view)

    return top_down_view


class ThorViz(TrajectoryViz):
    def __init__(
        self,
        path_to_trajectory: Sequence[str] = ("task_info", "followed_path"),
        label: str = "thor_trajectory",
        figsize: Tuple[int, int] = (4, 2),  # width, height
        fontsize: int = 5,
        scenes: Union[
            Tuple[str, int, int, int, int], Sequence[Tuple[str, int, int, int, int]]
        ] = ("FloorPlan_Val{}_{}", 1, 3, 1, 5),
        room_path: Sequence[str] = ("rl_robothor", "data", "topdown"),
        viz_rows_cols: Tuple[int, int] = (448, 448),
        single_color: bool = False,
        view_triangle_only_on_last: bool = True,
        disable_view_triangle: bool = False,
        line_opacity: float = 1.0,
    ):
        super().__init__(
            path_to_trajectory=path_to_trajectory,
            label=label,
            figsize=figsize,
            fontsize=fontsize,
        )

        if isinstance(scenes[0], str):
            scenes = [scenes]  # make it list of tuples
        self.scenes = scenes

        self.room_path = os.path.join(*room_path)
        self.viz_rows_cols = viz_rows_cols
        self.single_color = single_color
        self.view_triangle_only_on_last = view_triangle_only_on_last
        self.disable_view_triangle = disable_view_triangle
        self.line_opacity = line_opacity

        # Only needed for rendering
        self.map_data: Optional[Dict[str, Any]] = None
        self.thor_top_downs: Optional[Dict[str, np.ndarray]] = None

    def init_top_down_render(self):
        cont = Controller()
        self.map_data = self.get_translator(self.scenes, cont)
        self.thor_top_downs = self.make_top_down_views(self.scenes, cont)
        cont.stop()

    @staticmethod
    def iterate_scenes(all_scenes: Sequence[Tuple[str, int, int, int, int]]) -> str:
        for scenes in all_scenes:
            for wall in range(scenes[1], scenes[2] + 1):
                for furniture in range(scenes[3], scenes[4] + 1):
                    roomname = scenes[0].format(wall, furniture)
                    yield roomname

    def get_translator(
        self, all_scenes: Sequence[Tuple[str, int, int, int, int]], cont: Controller
    ) -> Dict[str, Any]:
        # TODO From the-robot-project (cached version):
        # map_data = {"pos_translator": ThorPositionTo2DFrameTranslator(
        #     self.viz_rows_cols,  # resolution (rows, cols)
        #     (5.05, 7.614471, -5.663423),  # camera pos (x, y, z)
        #     6.261584,  #
        # )}

        roomname = list(ThorViz.iterate_scenes(all_scenes))[0]
        cont.reset(roomname)
        map_data = get_agent_map_data(cont, self.viz_rows_cols)
        get_logger().debug("Using map_data {}".format(map_data))
        return map_data

    def make_top_down_views(
        self, all_scenes: Sequence[Tuple[str, int, int, int, int]], cont: Controller
    ) -> Dict[str, np.ndarray]:
        cont.step({"action": "ChangeQuality", "quality": "Very High"})
        cont.step({"action": "ChangeResolution", "x": 448, "y": 448})

        top_downs = {}
        for roomname in self.iterate_scenes(all_scenes):
            fname = os.path.join(self.room_path, "{}.png".format(roomname))
            if not os.path.exists(fname):
                top_downs[roomname] = dump_top_down_view(cont, roomname, self.room_path)
            else:
                top_downs[roomname] = cv2.imread(fname)

        return top_downs

    def crop_viz_image(self, viz_image: np.ndarray) -> np.ndarray:
        y_min = 0
        y_max = int(self.viz_rows_cols[0] * 0.5)
        x_min = 0
        x_max = self.viz_rows_cols[1]
        cropped_viz_image = viz_image[y_min:y_max, x_min:x_max, :]
        return cropped_viz_image

    def make_fig(self, episode: Any, episode_id: str) -> Figure:
        trajectory: Sequence[Dict[str:Any]] = self._access(
            episode, self.path_to_trajectory
        )

        if self.thor_top_downs is None:
            self.init_top_down_render()

        roomname = "FloorPlan_Val{}_{}".format(
            *episode_id.split("_")[1:3]
        )  # TODO HACK due to current episode id not including the full room name
        get_logger().debug("episode {} rommname {}".format(episode_id, roomname))

        im = visualize_agent_path(
            trajectory,
            self.thor_top_downs[roomname],
            self.map_data["pos_translator"],
            single_color=self.single_color,
            view_triangle_only_on_last=self.view_triangle_only_on_last,
            disable_view_triangle=self.disable_view_triangle,
            line_opacity=self.line_opacity,
        )

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_title(episode_id, fontsize=self.fontsize)
        ax.imshow(self.crop_viz_image(im)[:, :, ::-1])
        ax.axis("off")

        return fig
