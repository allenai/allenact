import copy
import json
import math
import os
from pathlib import Path
from typing import Tuple, Sequence, Union, Dict, Optional, Any, cast, Generator, List

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ai2thor.controller import Controller
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from utils.system import get_logger
from utils.viz_utils import TrajectoryViz

ROBOTHOR_VIZ_CACHED_TOPDOWN_VIEWS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(Path(__file__)), "data", "topdown")
)


class ThorPositionTo2DFrameTranslator(object):
    def __init__(
        self,
        frame_shape_rows_cols: Tuple[int, int],
        cam_position: Sequence[float],
        orth_size: float,
    ):
        self.frame_shape = frame_shape_rows_cols
        self.lower_left = np.array((cam_position[0], cam_position[2])) - orth_size
        self.span = 2 * orth_size

    def __call__(self, position: Sequence[float]):
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


class ThorViz(TrajectoryViz):
    def __init__(
        self,
        path_to_trajectory: Sequence[str] = ("task_info", "followed_path"),
        label: str = "thor_trajectory",
        figsize: Tuple[float, float] = (8, 4),  # width, height
        fontsize: float = 10,
        scenes: Union[
            Tuple[str, int, int, int, int], Sequence[Tuple[str, int, int, int, int]]
        ] = ("FloorPlan_Val{}_{}", 1, 3, 1, 5),
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
            scenes = [
                cast(Tuple[str, int, int, int, int], scenes)
            ]  # make it list of tuples
        self.scenes = cast(List[Tuple[str, int, int, int, int]], scenes)

        self.room_path = ROBOTHOR_VIZ_CACHED_TOPDOWN_VIEWS_DIR
        os.makedirs(self.room_path, exist_ok=True)

        self.viz_rows_cols = viz_rows_cols
        self.single_color = single_color
        self.view_triangle_only_on_last = view_triangle_only_on_last
        self.disable_view_triangle = disable_view_triangle
        self.line_opacity = line_opacity

        # Only needed for rendering
        self.map_data: Optional[Dict[str, Any]] = None
        self.thor_top_downs: Optional[Dict[str, np.ndarray]] = None

        self.controller: Optional[Controller] = None

    def init_top_down_render(self):
        self.map_data = self.get_translator()
        self.thor_top_downs = self.make_top_down_views()

        # No controller needed after this point
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    @staticmethod
    def iterate_scenes(
        all_scenes: Sequence[Tuple[str, int, int, int, int]]
    ) -> Generator[str, None, None]:
        for scenes in all_scenes:
            for wall in range(scenes[1], scenes[2] + 1):
                for furniture in range(scenes[3], scenes[4] + 1):
                    roomname = scenes[0].format(wall, furniture)
                    yield roomname

    def cached_map_data_path(self, roomname: str) -> str:
        return os.path.join(self.room_path, "map_data__{}.json".format(roomname))

    def get_translator(self) -> Dict[str, Any]:
        roomname = list(ThorViz.iterate_scenes(self.scenes))[0]
        json_file = self.cached_map_data_path(roomname)
        if not os.path.exists(json_file):
            self.make_controller()
            self.controller.reset(roomname)
            map_data = self.get_agent_map_data()
            get_logger().info("Dumping {}".format(json_file))
            with open(json_file, "w") as f:
                json.dump(map_data, f, indent=4, sort_keys=True)
        else:
            with open(json_file, "r") as f:
                map_data = json.load(f)

        pos_translator = ThorPositionTo2DFrameTranslator(
            self.viz_rows_cols,
            self.position_to_tuple(map_data["cam_position"]),
            map_data["cam_orth_size"],
        )
        map_data["pos_translator"] = pos_translator

        get_logger().debug("Using map_data {}".format(map_data))
        return map_data

    def cached_image_path(self, roomname: str) -> str:
        return os.path.join(
            self.room_path, "{}__r{}_c{}.png".format(roomname, *self.viz_rows_cols)
        )

    def make_top_down_views(self) -> Dict[str, np.ndarray]:
        top_downs = {}
        for roomname in self.iterate_scenes(self.scenes):
            fname = self.cached_image_path(roomname)
            if not os.path.exists(fname):
                self.make_controller()
                self.dump_top_down_view(roomname, fname)
            top_downs[roomname] = cv2.imread(fname)

        return top_downs

    def crop_viz_image(self, viz_image: np.ndarray) -> np.ndarray:
        # Top-down view of room spans vertically near the center of the frame in RoboTHOR:
        y_min = int(self.viz_rows_cols[0] * 0.3)
        y_max = int(self.viz_rows_cols[0] * 0.8)
        # But it covers approximately the entire width:
        x_min = 0
        x_max = self.viz_rows_cols[1]
        cropped_viz_image = viz_image[y_min:y_max, x_min:x_max, :]
        return cropped_viz_image

    def make_controller(self):
        if self.controller is None:
            self.controller = Controller()

            self.controller.step({"action": "ChangeQuality", "quality": "Very High"})
            self.controller.step(
                {
                    "action": "ChangeResolution",
                    "x": self.viz_rows_cols[1],
                    "y": self.viz_rows_cols[0],
                }
            )

    def get_agent_map_data(self):
        self.controller.step({"action": "ToggleMapView"})
        cam_position = self.controller.last_event.metadata["cameraPosition"]
        cam_orth_size = self.controller.last_event.metadata["cameraOrthSize"]
        to_return = {
            "cam_position": cam_position,
            "cam_orth_size": cam_orth_size,
        }
        self.controller.step({"action": "ToggleMapView"})
        return to_return

    @staticmethod
    def position_to_tuple(position: Dict[str, float]) -> Tuple[float, float, float]:
        return position["x"], position["y"], position["z"]

    @staticmethod
    def add_lines_to_map(
        ps: Sequence[Any],
        frame: np.ndarray,
        pos_translator: ThorPositionTo2DFrameTranslator,
        opacity: float,
        color: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
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

    @staticmethod
    def add_line_to_map(
        p0: Any,
        p1: Any,
        frame: np.ndarray,
        pos_translator: ThorPositionTo2DFrameTranslator,
        opacity: float,
        color: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
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

    @staticmethod
    def add_agent_view_triangle(
        position: Any,
        rotation: Dict[str, float],
        frame: np.ndarray,
        pos_translator: ThorPositionTo2DFrameTranslator,
        scale: float = 1.0,
        opacity: float = 0.1,
    ) -> np.ndarray:
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

    @staticmethod
    def visualize_agent_path(
        positions: Sequence[Any],
        frame: np.ndarray,
        pos_translator: ThorPositionTo2DFrameTranslator,
        single_color: bool = False,
        view_triangle_only_on_last: bool = False,
        disable_view_triangle: bool = False,
        line_opacity: float = 1.0,
    ) -> np.ndarray:
        if single_color:
            frame = ThorViz.add_lines_to_map(
                list(map(ThorViz.position_to_tuple, positions)),
                frame,
                pos_translator,
                line_opacity,
                (0, 255, 0),
            )
        else:
            import colour as col

            colors = list(
                col.Color("red").range_to(col.Color("green"), len(positions) - 1)
            )
            for i in range(len(positions) - 1):
                frame = ThorViz.add_line_to_map(
                    ThorViz.position_to_tuple(positions[i]),
                    ThorViz.position_to_tuple(positions[i + 1]),
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
            frame = ThorViz.add_agent_view_triangle(
                ThorViz.position_to_tuple(position),
                rotation=position["rotation"],
                frame=frame,
                pos_translator=pos_translator,
                opacity=0.05 + view_triangle_only_on_last * 0.2,
            )
        return frame

    def dump_top_down_view(self, room_name: str, image_path: str):
        get_logger().debug("Dumping {}".format(image_path))

        self.controller.reset(room_name)
        self.controller.step(
            {"action": "Initialize", "gridSize": 0.1, "makeAgentsVisible": False}
        )
        self.controller.step({"action": "ToggleMapView"})
        top_down_view = self.controller.last_event.cv2img

        cv2.imwrite(image_path, top_down_view)

    def make_fig(self, episode: Any, episode_id: str) -> Figure:
        trajectory: Sequence[Dict[str, Any]] = self._access(
            episode, self.path_to_trajectory
        )

        if self.thor_top_downs is None:
            self.init_top_down_render()

        roomname = "_".join(episode_id.split("_")[:3])

        im = self.visualize_agent_path(
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
