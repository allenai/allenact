"""Utility functions and classes for visualization and logging."""

import os
from datetime import datetime

import cv2
import imageio
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from allenact_plugins.manipulathor_plugin.manipulathor_utils import initialize_arm
from allenact_plugins.manipulathor_plugin.manipulathor_utils import (
    reset_environment_and_additional_commands,
    transport_wrapper,
)


class LoggerVisualizer:
    def __init__(self, exp_name="", log_dir=""):
        if log_dir == "":
            log_dir = self.__class__.__name__
        if exp_name == "":
            exp_name = "NoNameExp"
        self.exp_name = exp_name
        log_dir = os.path.join(
            exp_name,
            log_dir,
        )
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_queue = []
        self.action_queue = []
        self.logger_index = 0

    def log(self, environment, action_str):
        raise Exception("Not Implemented")

    def is_empty(self):
        return len(self.log_queue) == 0

    def finish_episode_metrics(self, episode_info, task_info, metric_results):
        pass

    def finish_episode(self, environment, episode_info, task_info):
        pass


class TestMetricLogger(LoggerVisualizer):
    def __init__(self, exp_name="", log_dir="", **kwargs):
        super().__init__(exp_name=exp_name, log_dir=log_dir)
        self.total_metric_dict = {}
        log_file_name = os.path.join(self.log_dir, "test_metric.txt")
        self.metric_log_file = open(log_file_name, "w")
        self.disturbance_distance_queue = []

    def average_dict(self):
        result = {}
        for k, v in self.total_metric_dict.items():
            result[k] = sum(v) / len(v)
        return result

    def finish_episode_metrics(self, episode_info, task_info, metric_results=None):

        if metric_results is None:
            print("had to reset")
            self.action_queue = []
            self.disturbance_distance_queue = []
            return

        for k in metric_results.keys():
            if "metric" in k or k in ["ep_length", "reward", "success"]:
                self.total_metric_dict.setdefault(k, [])
                self.total_metric_dict[k].append(metric_results[k])
        print(
            "total",
            len(self.total_metric_dict["success"]),
            "average test metric",
            self.average_dict(),
        )

        # save the task info and all the action queue and results
        log_dict = {
            "logger_number": self.logger_index,
            "action_sequence": self.action_queue,
            "disturbance_sequence": self.disturbance_distance_queue,
            "task_info_metrics": metric_results,
        }
        self.logger_index += 1
        self.metric_log_file.write(str(log_dict))
        self.metric_log_file.write("\n")
        self.metric_log_file.flush()
        print("Logging to", self.metric_log_file.name)

        self.action_queue = []
        self.disturbance_distance_queue = []

    def log(self, environment, action_str="", disturbance_str=""):
        # We can add agent arm and state location if needed
        self.action_queue.append(action_str)
        self.disturbance_distance_queue.append(disturbance_str)


class BringObjImageVisualizer(LoggerVisualizer):
    def finish_episode(self, environment, episode_info, task_info):
        now = datetime.now()
        time_to_write = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
        time_to_write += "log_ind_{}".format(self.logger_index)
        self.logger_index += 1
        print("Loggigng", time_to_write, "len", len(self.log_queue))

        source_object_id = task_info["source_object_id"]
        goal_object_id = task_info["goal_object_id"]
        pickup_success = episode_info.object_picked_up
        episode_success = episode_info._success

        # Put back if you want the images
        # for i, img in enumerate(self.log_queue):
        #     image_dir = os.path.join(self.log_dir, time_to_write + '_seq{}.png'.format(str(i)))
        #     cv2.imwrite(image_dir, img[:,:,[2,1,0]])

        episode_success_offset = "succ" if episode_success else "fail"
        pickup_success_offset = "succ" if pickup_success else "fail"

        gif_name = (
            time_to_write
            + "_from_"
            + source_object_id.split("|")[0]
            + "_to_"
            + goal_object_id.split("|")[0]
            + "_pickup_"
            + pickup_success_offset
            + "_episode_"
            + episode_success_offset
            + ".gif"
        )
        concat_all_images = np.expand_dims(np.stack(self.log_queue, axis=0), axis=1)
        save_image_list_to_gif(concat_all_images, gif_name, self.log_dir)
        this_controller = environment.controller
        scene = this_controller.last_event.metadata["sceneName"]
        reset_environment_and_additional_commands(this_controller, scene)
        self.log_start_goal(
            environment,
            task_info["visualization_source"],
            tag="start",
            img_adr=os.path.join(self.log_dir, time_to_write),
        )
        self.log_start_goal(
            environment,
            task_info["visualization_target"],
            tag="goal",
            img_adr=os.path.join(self.log_dir, time_to_write),
        )

        self.log_queue = []
        self.action_queue = []

    def log(self, environment, action_str):
        image_tensor = environment.current_frame
        self.action_queue.append(action_str)
        self.log_queue.append(image_tensor)

    def log_start_goal(self, env, task_info, tag, img_adr):
        object_location = task_info["object_location"]
        object_id = task_info["object_id"]
        agent_state = task_info["agent_pose"]
        this_controller = env.controller
        # We should not reset here
        # for start arm from high up as a cheating, this block is very important. never remove
        event1, event2, event3 = initialize_arm(this_controller)
        if not (
            event1.metadata["lastActionSuccess"]
            and event2.metadata["lastActionSuccess"]
            and event3.metadata["lastActionSuccess"]
        ):
            print("ERROR: ARM MOVEMENT FAILED in logging! SHOULD NEVER HAPPEN")

        event = transport_wrapper(this_controller, object_id, object_location)
        if not event.metadata["lastActionSuccess"]:
            print("ERROR: oh no could not transport in logging")

        event = this_controller.step(
            dict(
                action="TeleportFull",
                standing=True,
                x=agent_state["position"]["x"],
                y=agent_state["position"]["y"],
                z=agent_state["position"]["z"],
                rotation=dict(
                    x=agent_state["rotation"]["x"],
                    y=agent_state["rotation"]["y"],
                    z=agent_state["rotation"]["z"],
                ),
                horizon=agent_state["cameraHorizon"],
            )
        )
        if not event.metadata["lastActionSuccess"]:
            print("ERROR: oh no could not teleport in logging")

        image_tensor = this_controller.last_event.frame
        image_dir = (
            img_adr + "_obj_" + object_id.split("|")[0] + "_pickup_" + tag + ".png"
        )
        cv2.imwrite(image_dir, image_tensor[:, :, [2, 1, 0]])

        # Saving the mask
        target_object_id = task_info["object_id"]
        all_visible_masks = this_controller.last_event.instance_masks
        if target_object_id in all_visible_masks:
            mask_frame = all_visible_masks[target_object_id]
        else:
            mask_frame = np.zeros(env.controller.last_event.frame[:, :, 0].shape)
        mask_dir = (
            img_adr + "_obj_" + object_id.split("|")[0] + "_pickup_" + tag + "_mask.png"
        )
        cv2.imwrite(mask_dir, mask_frame.astype(float) * 255.0)


class ImageVisualizer(LoggerVisualizer):
    def __init__(
        self,
        exp_name="",
        log_dir="",
        add_top_down_view: bool = False,
        add_depth_map: bool = False,
    ):
        super().__init__(exp_name=exp_name, log_dir=log_dir)
        self.add_top_down_view = add_top_down_view
        self.add_depth_map = add_depth_map
        if self.add_top_down_view:
            self.top_down_queue = []
        self.disturbance_distance_queue = []

    def finish_episode(self, environment, episode_info, task_info):
        time_to_write = "log_ind_{:03d}".format(self.logger_index)
        self.logger_index += 1
        print("Logging", time_to_write, "len", len(self.log_queue))
        object_id = task_info["objectId"]
        scene_name = task_info["source_location"]["scene_name"]
        source_countertop = task_info["source_location"]["countertop_id"]
        target_countertop = task_info["target_location"]["countertop_id"]

        pickup_success = episode_info.object_picked_up
        episode_success = episode_info._success

        # Put back if you want the images
        # for i, img in enumerate(self.log_queue):
        #     image_dir = os.path.join(self.log_dir, time_to_write + '_seq{}.png'.format(str(i)))
        #     cv2.imwrite(image_dir, img[:,:,[2,1,0]])

        episode_success_offset = "succ" if episode_success else "fail"
        pickup_success_offset = "succ" if pickup_success else "fail"
        gif_name = (
            time_to_write
            + "_pickup_"
            + pickup_success_offset
            + "_episode_"
            + episode_success_offset
            + "_"
            + scene_name.split("_")[0]
            + "_obj_"
            + object_id.split("|")[0]
            + "_from_"
            + source_countertop.split("|")[0]
            + "_to_"
            + target_countertop.split("|")[0]
            + ".gif"
        )

        self.log_queue = put_annotation_on_image(
            self.log_queue, self.disturbance_distance_queue
        )

        concat_all_images = np.expand_dims(np.stack(self.log_queue, axis=0), axis=1)
        if self.add_top_down_view:
            topdown_all_images = np.expand_dims(
                np.stack(self.top_down_queue, axis=0), axis=1
            )  # (T, 1, H, W, 3)
            concat_all_images = np.concatenate(
                [concat_all_images, topdown_all_images], axis=1
            )  # (T, 2, H, W, 3)

        save_image_list_to_gif(concat_all_images, gif_name, self.log_dir)

        self.log_start_goal(
            environment,
            task_info["visualization_source"],
            tag="start",
            img_adr=os.path.join(self.log_dir, time_to_write),
        )
        self.log_start_goal(
            environment,
            task_info["visualization_target"],
            tag="goal",
            img_adr=os.path.join(self.log_dir, time_to_write),
        )

        self.log_queue = []
        self.action_queue = []
        self.disturbance_distance_queue = []
        if self.add_top_down_view:
            self.top_down_queue = []

    def log(self, environment, action_str="", disturbance_str=""):
        self.action_queue.append(action_str)
        self.disturbance_distance_queue.append(disturbance_str)

        image_tensor = environment.current_frame
        self.log_queue.append(image_tensor)

        if self.add_top_down_view:
            # Reference: https://github.com/allenai/ai2thor/pull/814
            event = environment.controller.step(action="GetMapViewCameraProperties")
            event = environment.controller.step(
                action="AddThirdPartyCamera", **event.metadata["actionReturn"]
            )
            self.top_down_queue.append(event.third_party_camera_frames[0])

    def log_start_goal(self, env, task_info, tag, img_adr):
        object_location = task_info["object_location"]
        object_id = task_info["object_id"]
        agent_state = task_info["agent_pose"]
        this_controller = env.controller
        scene = this_controller.last_event.metadata[
            "sceneName"
        ]  # maybe we need to reset env actually]
        reset_environment_and_additional_commands(this_controller, scene)
        # for start arm from high up as a cheating, this block is very important. never remove
        event1, event2, event3 = initialize_arm(this_controller)
        if not (
            event1.metadata["lastActionSuccess"]
            and event2.metadata["lastActionSuccess"]
            and event3.metadata["lastActionSuccess"]
        ):
            print("ERROR: ARM MOVEMENT FAILED in logging! SHOULD NEVER HAPPEN")

        event = transport_wrapper(this_controller, object_id, object_location)
        if not event.metadata["lastActionSuccess"]:
            print("ERROR: oh no could not transport in logging")

        event = this_controller.step(
            dict(
                action="TeleportFull",
                standing=True,
                x=agent_state["position"]["x"],
                y=agent_state["position"]["y"],
                z=agent_state["position"]["z"],
                rotation=dict(
                    x=agent_state["rotation"]["x"],
                    y=agent_state["rotation"]["y"],
                    z=agent_state["rotation"]["z"],
                ),
                horizon=agent_state["cameraHorizon"],
            )
        )
        if not event.metadata["lastActionSuccess"]:
            print("ERROR: oh no could not teleport in logging")

        image_tensor = this_controller.last_event.frame
        image_dir = img_adr + "_" + tag + ".png"
        cv2.imwrite(image_dir, image_tensor[:, :, [2, 1, 0]])

        if self.add_depth_map:
            depth = this_controller.last_event.depth_frame.copy()  # (H, W)
            depth[depth > 5.0] = 5.0
            norm = matplotlib.colors.Normalize(vmin=depth.min(), vmax=depth.max())
            rgb = cm.get_cmap(plt.get_cmap("viridis"))(norm(depth))[:, :, :3]  # [0,1]
            rgb = (rgb * 255).astype(np.uint8)

            depth_dir = img_adr + "_" + tag + "_depth.png"
            cv2.imwrite(depth_dir, rgb[:, :, [2, 1, 0]])


def save_image_list_to_gif(image_list, gif_name, gif_dir):
    gif_adr = os.path.join(gif_dir, gif_name)

    seq_len, cols, w, h, c = image_list.shape

    pallet = np.zeros(
        (seq_len, w, h * cols, c)
    )  # to support multiple animations in one gif

    for col_ind in range(cols):
        pallet[:, :, col_ind * h : (col_ind + 1) * h, :] = image_list[:, col_ind]

    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    imageio.mimsave(gif_adr, pallet.astype(np.uint8), format="GIF", duration=1 / 5)
    print("Saved result in ", gif_adr)


def put_annotation_on_image(images, annotations):
    all_images = []
    for img, annot in zip(images, annotations):
        position = (10, 10)

        from PIL import Image, ImageDraw

        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, annot, (0, 0, 0))
        all_images.append(np.array(pil_img))

    return all_images
