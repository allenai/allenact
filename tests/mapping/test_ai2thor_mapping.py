import os
import platform
import random
import sys
import urllib
import urllib.request
import warnings
from collections import defaultdict
# noinspection PyUnresolvedReferences
from tempfile import mkdtemp
from typing import Dict, List, Tuple, cast

# noinspection PyUnresolvedReferences
import ai2thor
# noinspection PyUnresolvedReferences
import ai2thor.wsgi_server
import compress_pickle
import numpy as np
import torch

from allenact.algorithms.onpolicy_sync.storage import RolloutBlockStorage
from allenact.base_abstractions.misc import Memory, ActorCriticOutput
from allenact.embodiedai.mapping.mapping_utils.map_builders import SemanticMapBuilder
from allenact.utils.experiment_utils import set_seed
from allenact.utils.system import get_logger
from allenact.utils.tensor_utils import batch_observations
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RelativePositionChangeTHORSensor,
    ReachableBoundsTHORSensor,
    BinnedPointCloudMapTHORSensor,
    SemanticMapTHORSensor,
)
from allenact_plugins.ithor_plugin.ithor_util import get_open_x_displays
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from constants import ABS_PATH_OF_TOP_LEVEL_DIR


class TestAI2THORMapSensors(object):
    def setup_path_for_use_with_rearrangement_project(self) -> bool:
        if platform.system() != "Darwin" and len(get_open_x_displays()) == 0:
            wrn_msg = "Cannot run tests as there seem to be no open displays!"
            warnings.warn(wrn_msg)
            get_logger().warning(wrn_msg)
            return False

        os.chdir(ABS_PATH_OF_TOP_LEVEL_DIR)
        sys.path.append(
            os.path.join(ABS_PATH_OF_TOP_LEVEL_DIR, "projects/ithor_rearrangement")
        )
        try:
            import rearrange
        except ImportError:
            wrn_msg = (
                "Could not import `rearrange`. Is it possible you have"
                " not initialized the submodules (i.e. by running"
                " `git submodule init; git submodule update;`)?"
            )
            warnings.warn(wrn_msg)
            get_logger().warning(wrn_msg)
            return False

        return True

    def test_binned_and_semantic_mapping(self, tmpdir):
        try:
            if not self.setup_path_for_use_with_rearrangement_project():
                return

            from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
            from baseline_configs.walkthrough.walkthrough_rgb_base import (
                WalkthroughBaseExperimentConfig,
            )
            from rearrange.constants import (
                FOV,
                PICKUPABLE_OBJECTS,
                OPENABLE_OBJECTS,
            )
            from datagen.datagen_utils import get_scenes

            ORDERED_OBJECT_TYPES = list(sorted(PICKUPABLE_OBJECTS + OPENABLE_OBJECTS))

            map_range_sensor = ReachableBoundsTHORSensor(margin=1.0)
            map_info = dict(
                map_range_sensor=map_range_sensor,
                vision_range_in_cm=40 * 5,
                map_size_in_cm=1050,
                resolution_in_cm=5,
            )
            map_sensors = [
                RelativePositionChangeTHORSensor(),
                map_range_sensor,
                DepthSensorThor(
                    height=224, width=224, use_normalization=False, uuid="depth",
                ),
                BinnedPointCloudMapTHORSensor(fov=FOV, ego_only=False, **map_info,),
                SemanticMapTHORSensor(
                    fov=FOV,
                    ego_only=False,
                    ordered_object_types=ORDERED_OBJECT_TYPES,
                    **map_info,
                ),
            ]
            all_sensors = [*WalkthroughBaseExperimentConfig.SENSORS, *map_sensors]

            open_x_displays = []
            try:
                open_x_displays = get_open_x_displays()
            except (AssertionError, IOError):
                pass
            walkthrough_task_sampler = WalkthroughBaseExperimentConfig.make_sampler_fn(
                stage="train",
                sensors=all_sensors,
                scene_to_allowed_rearrange_inds={s: [0] for s in get_scenes("train")},
                force_cache_reset=True,
                allowed_scenes=None,
                seed=1,
                x_display=open_x_displays[0] if len(open_x_displays) != 0 else None,
                thor_controller_kwargs={
                    **RearrangeBaseExperimentConfig.THOR_CONTROLLER_KWARGS,
                    # "server_class": ai2thor.wsgi_server.WsgiServer,  # Only for debugging
                },
            )

            targets_path = os.path.join(tmpdir, "rearrange_mapping_examples.pkl.gz")
            urllib.request.urlretrieve(
                "https://ai2-prior-allenact-public-test.s3-us-west-2.amazonaws.com/ai2thor_mapping/rearrange_mapping_examples.pkl.gz",
                targets_path,
            )
            goal_obs_dict = compress_pickle.load(targets_path)

            def compare_recursive(obs, goal_obs, key_list: List):
                if isinstance(obs, Dict):
                    for k in goal_obs:
                        compare_recursive(
                            obs=obs[k], goal_obs=goal_obs[k], key_list=key_list + [k]
                        )
                elif isinstance(obs, (List, Tuple)):
                    for i in range(len(goal_obs)):
                        compare_recursive(
                            obs=obs[i], goal_obs=goal_obs[i], key_list=key_list + [i]
                        )
                else:
                    # Should be a numpy array at this point
                    assert isinstance(obs, np.ndarray) and isinstance(
                        goal_obs, np.ndarray
                    ), f"After {key_list}, not numpy arrays, obs={obs}, goal_obs={goal_obs}"

                    obs = 1.0 * obs
                    goal_obs = 1.0 * goal_obs

                    where_nan = np.isnan(goal_obs)
                    obs[where_nan] = 0.0
                    goal_obs[where_nan] = 0.0
                    assert (
                        np.abs(1.0 * obs - 1.0 * goal_obs).mean() < 1e-4
                    ), f"Difference of {np.abs(1.0 * obs - 1.0 * goal_obs).mean()} at {key_list}."

            observations_dict = defaultdict(lambda: [])
            for i in range(5):  # Why 5, why not 5?
                set_seed(i)
                task = walkthrough_task_sampler.next_task()

                obs_list = observations_dict[i]
                obs_list.append(task.get_observations())
                k = 0
                compare_recursive(
                    obs=obs_list[0], goal_obs=goal_obs_dict[i][0], key_list=[i, k]
                )
                while not task.is_done():
                    obs = task.step(
                        action=task.action_names().index(
                            random.choice(
                                3
                                * [
                                    "move_ahead",
                                    "rotate_right",
                                    "rotate_left",
                                    "look_up",
                                    "look_down",
                                ]
                                + ["done"]
                            )
                        )
                    ).observation
                    k += 1
                    obs_list.append(obs)
                    compare_recursive(
                        obs=obs,
                        goal_obs=goal_obs_dict[i][task.num_steps_taken()],
                        key_list=[i, k],
                    )

                    # Free space metric map in RGB using pointclouds coming from depth images. This
                    # is built iteratively after every step.
                    # R - is used to encode points at a height < 0.02m (i.e. the floor)
                    # G - is used to encode points at a height between 0.02m and 2m, i.e. objects the agent would run into
                    # B - is used to encode points higher than 2m, i.e. ceiling

                    # Uncomment if you wish to visualize the observations:
                    # import matplotlib.pyplot as plt
                    # plt.imshow(
                    #     np.flip(255 * (obs["binned_pc_map"]["map"] > 0), 0)
                    # )  # np.flip because we expect "up" to be -row
                    # plt.title("Free space map")
                    # plt.show()
                    # plt.close()

                    # See also `obs["binned_pc_map"]["egocentric_update"]` to see the
                    # the metric map from the point of view of the agent before it is
                    # rotated into the world-space coordinates and merged with past observations.

                    # Semantic map in RGB which is iteratively revealed using depth maps to figure out what
                    # parts of the scene the agent has seen so far.
                    # This map has shape 210x210x72 with the 72 channels corresponding to the 72
                    # object types in `ORDERED_OBJECT_TYPES`
                    semantic_map = obs["semantic_map"]["map"]

                    # We can't display all 72 channels in an RGB image so instead we randomly assign
                    # each object a color and then just allow them to overlap each other
                    colored_semantic_map = SemanticMapBuilder.randomly_color_semantic_map(
                        semantic_map
                    )

                    # Here's the full semantic map with nothing masked out because the agent
                    # hasn't seen it yet
                    colored_semantic_map_no_fog = SemanticMapBuilder.randomly_color_semantic_map(
                        map_sensors[-1].semantic_map_builder.ground_truth_semantic_map
                    )

                    # Uncomment if you wish to visualize the observations:
                    # import matplotlib.pyplot as plt
                    # plt.imshow(
                    #     np.flip(  # np.flip because we expect "up" to be -row
                    #         np.concatenate(
                    #             (
                    #                 colored_semantic_map,
                    #                 255 + 0 * colored_semantic_map[:, :10, :],
                    #                 colored_semantic_map_no_fog,
                    #             ),
                    #             axis=1,
                    #         ),
                    #         0,
                    #     )
                    # )
                    # plt.title("Semantic map with and without exploration fog")
                    # plt.show()
                    # plt.close()

                    # See also
                    # * `obs["semantic_map"]["egocentric_update"]`
                    # * `obs["semantic_map"]["explored_mask"]`
                    # * `obs["semantic_map"]["egocentric_mask"]`

            # To save observations for comparison against future runs, uncomment the below.
            # os.makedirs("tmp_out", exist_ok=True)
            # compress_pickle.dump(
            #     {**observations_dict}, "tmp_out/rearrange_mapping_examples.pkl.gz"
            # )
        finally:
            try:
                walkthrough_task_sampler.close()
            except NameError:
                pass

    def test_pretrained_rearrange_walkthrough_mapping_agent(self, tmpdir):
        try:
            if not self.setup_path_for_use_with_rearrangement_project():
                return

            from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
            from baseline_configs.walkthrough.walkthrough_rgb_mapping_ppo import (
                WalkthroughRGBMappingPPOExperimentConfig,
            )
            from rearrange.constants import (
                FOV,
                PICKUPABLE_OBJECTS,
                OPENABLE_OBJECTS,
            )
            from datagen.datagen_utils import get_scenes

            open_x_displays = []
            try:
                open_x_displays = get_open_x_displays()
            except (AssertionError, IOError):
                pass
            walkthrough_task_sampler = WalkthroughRGBMappingPPOExperimentConfig.make_sampler_fn(
                stage="train",
                scene_to_allowed_rearrange_inds={s: [0] for s in get_scenes("train")},
                force_cache_reset=True,
                allowed_scenes=None,
                seed=2,
                x_display=open_x_displays[0] if len(open_x_displays) != 0 else None,
            )

            named_losses = (
                WalkthroughRGBMappingPPOExperimentConfig.training_pipeline().named_losses
            )

            ckpt_path = os.path.join(
                tmpdir, "pretrained_walkthrough_mapping_agent_75mil.pt"
            )
            if not os.path.exists(ckpt_path):
                urllib.request.urlretrieve(
                    "https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/rearrangement/walkthrough/pretrained_walkthrough_mapping_agent_75mil.pt",
                    ckpt_path,
                )

            state_dict = torch.load(ckpt_path, map_location="cpu",)

            walkthrough_model = WalkthroughRGBMappingPPOExperimentConfig.create_model()
            walkthrough_model.load_state_dict(state_dict["model_state_dict"])

            rollout_storage = RolloutBlockStorage(
                init_size=1,
                num_samplers=1,
                actor_critic=walkthrough_model,
            )
            memory = rollout_storage.pick_memory_step(0)
            masks = rollout_storage.masks[:1]

            binned_map_losses = []
            semantic_map_losses = []
            for i in range(5):
                masks = 0 * masks

                set_seed(i + 1)
                task = walkthrough_task_sampler.next_task()

                def add_step_dim(input):
                    if isinstance(input, torch.Tensor):
                        return input.unsqueeze(0)
                    elif isinstance(input, Dict):
                        return {k: add_step_dim(v) for k, v in input.items()}
                    else:
                        raise NotImplementedError

                batch = add_step_dim(batch_observations([task.get_observations()]))

                while not task.is_done():
                    ac_out, memory = cast(
                        Tuple[ActorCriticOutput, Memory],
                        walkthrough_model.forward(
                            observations=batch,
                            memory=memory,
                            prev_actions=None,
                            masks=masks,
                        ),
                    )

                    binned_map_losses.append(
                        named_losses["binned_map_loss"]
                        .loss(
                            step_count=0,  # Not used in this loss
                            batch={"observations": batch},
                            actor_critic_output=ac_out,
                        )[0]
                        .item()
                    )
                    assert (
                        binned_map_losses[-1] < 0.16
                    ), f"Binned map loss to large at ({i}, {task.num_steps_taken()})"

                    semantic_map_losses.append(
                        named_losses["semantic_map_loss"]
                        .loss(
                            step_count=0,  # Not used in this loss
                            batch={"observations": batch},
                            actor_critic_output=ac_out,
                        )[0]
                        .item()
                    )
                    assert (
                        semantic_map_losses[-1] < 0.004
                    ), f"Semantic map loss to large at ({i}, {task.num_steps_taken()})"

                    masks = masks.fill_(1.0)
                    obs = task.step(
                        action=ac_out.distributions.sample().item()
                    ).observation
                    batch = add_step_dim(batch_observations([obs]))

                    if task.num_steps_taken() >= 10:
                        break

            # To save observations for comparison against future runs, uncomment the below.
            # os.makedirs("tmp_out", exist_ok=True)
            # compress_pickle.dump(
            #     {**observations_dict}, "tmp_out/rearrange_mapping_examples.pkl.gz"
            # )
        finally:
            try:
                walkthrough_task_sampler.close()
            except NameError:
                pass


if __name__ == "__main__":
    TestAI2THORMapSensors().test_binned_and_semantic_mapping(mkdtemp())  # type:ignore
    # TestAI2THORMapSensors().test_binned_and_semantic_mapping("tmp_out")  # Used for local debugging
    # TestAI2THORMapSensors().test_pretrained_rearrange_walkthrough_mapping_agent(
    #     "tmp_out"
    # )  # Used for local debugging
