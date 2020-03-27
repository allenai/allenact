from invoke import task

from utils.system import init_logging, LOGGER
from utils.experiment_utils import set_seed

# @task
# def create_robothor_pointnav_dataset(ctxt, rotateStepDegrees=45.0, visibilityDistance=1.0, gridSize=0.25,
#                                      samples_per_scene=4000, initials_per_target = 50, width=300, height=300,
#                                      fieldOfView=79, num_processes=32, num_gpus=8):
#     scenes = [
#         "FloorPlan_Train%d_%d" % (wall, furniture)
#         for wall in range(1, 13)
#         for furniture in range(1, 6)
#     ]
#
#     inputs = []
#     it = 0
#     for scene in scenes:
#         inputs.append((it % num_gpus, scene, rotateStepDegrees, visibilityDistance, gridSize, samples_per_scene,
#                        initials_per_target, width, height, fieldOfView))
#         it += 1
#
#     pool = mp.Pool(processes=num_processes)
#     pool.map(create_robothor_pointnav_dataset_worker, inputs)
#
#     pool.close()
#     pool.join()
#
#
# def create_robothor_pointnav_dataset_worker(inputs):
#     gpu_id, scene, rotateStepDegrees, visibilityDistance, gridSize, samples_per_scene, initials_per_target, width, height, fieldOfView = inputs
#     from rl_robothor.robothor_environment import RoboThorEnvironment as Env
#     import random
#     import numpy as np
#     import copy
#     from ai2thor.util import metrics
#     import json
#
#     print(gpu_id, scene)
#
#     cfg = dict(
#         rotateStepDegrees=rotateStepDegrees,
#         visibilityDistance=visibilityDistance,
#         gridSize=gridSize,
#         width=width,
#         height=height,
#         fieldOfView=fieldOfView,
#         quality="Very Low",
#         continuousMode=True,
#         applyActionNoise=True,
#         agentType="stochastic",
#         snapToGrid=False,
#         agentMode="bot",
#         movementGaussianMu=1e-20,
#         movementGaussianSigma=1e-20,
#         rotateGaussianMu=1e-20,
#         rotateGaussianSigma=1e-20,
#         x_display="0.%d" % gpu_id
#     )
#     env = Env(**cfg)
#
#     def get_shortest_path_to_point(
#             controller,
#             initial_position,
#             target_position
#     ):
#         args = dict(
#             action='GetShortestPathToPoint',
#             position=initial_position,
#             x=target_position['x'],
#             y=target_position['y'],
#             z=target_position['z']
#         )
#         event = controller.step(args)
#         if event.metadata['lastActionSuccess']:
#             return event.metadata['actionReturn']['corners']
#         else:
#             raise ValueError(
#                 "Unable to find shortest path for target point '{}'".format(
#                     target_position
#                 )
#             )
#
#     def extract_targets(points, npoints, margin):
#         all_points = points
#         targets = []
#         while len(targets) < npoints:
#             valid = False
#             nattempts = 0
#             while not valid and nattempts < 10:
#                 tget = random.choice(points)
#                 # samps = random.choices(all_points, k=10)
#                 random.shuffle(all_points)
#                 samps = all_points[:10]
#                 for samp in samps:
#                     try:
#                         path = get_shortest_path_to_point(env.controller, samp, tget)
#                         valid = True
#                         break
#                     except ValueError:
#                         continue
#                 nattempts += 1
#             if valid:
#                 targets.append(tget)
#                 points = [p for p in points if (p['x'] - tget['x']) ** 2 + (p['z'] - tget['z']) ** 2 > margin * margin]
#             else:
#                 print("Stopping after 10 attempts with {} targets out of {} desired".format(len(targets), npoints))
#                 break
#         return targets
#
#     def extract_initials(points, npoints, margin, tget):
#         initial = []
#         # points = set([(p['x'], p['y'], p['z']) for p in points])
#         while len(initial) < npoints:
#             valid = False
#             nattempts = 0
#             while not valid and nattempts < 10:
#                 samp = random.choice(points)
#                 # dsamp = dict(x=samp[0], y=samp[1], z=samp[2])
#                 try:
#                     path = get_shortest_path_to_point(env.controller, samp, tget)
#                     valid = True
#                     break
#                 except ValueError:
#                     # points.remove(samp)
#                     points = [p for p in points if p != samp]
#                     continue
#                 nattempts += 1
#             if valid:
#                 initial.append(samp)
#                 # points = set([p for p in points if (p['x'] - samp['x']) ** 2 + (p['z'] - samp['z']) ** 2 > margin * margin])
#                 points = [p for p in points if (p['x'] - samp['x']) ** 2 + (p['z'] - samp['z']) ** 2 > margin * margin]
#             else:
#                 print("Stopping after 10 attempts with {} initial points out of {} desired".format(len(points), npoints))
#                 break
#         return initial
#
#     all_episodes = []
#     env.reset(scene)
#     candidates = copy.copy(env.currently_reachable_points)
#     print('{} candidates'.format(len(candidates)))
#     targets = extract_targets(copy.copy(candidates), npoints=samples_per_scene // initials_per_target, margin=2 * gridSize)
#     print('{} targets'.format(len(targets)))
#     scene_episodes = []
#     ntargets = 0
#     for target in targets:
#         initials = extract_initials(copy.copy(candidates), npoints=2 * initials_per_target, margin=0.5 * gridSize, tget=target)
#         random.shuffle(initials)
#         initials = initials[:initials_per_target]
#
#         ninitials = 0
#         eps = 0.0001
#         for pos_unity in initials:
#             possible_orientations = np.linspace(0, 360, num=round(360/rotateStepDegrees), endpoint=True).tolist()
#
#             try:
#                 # print('{} to {} in {}'.format(pos_unity, target, scene))
#                 path = get_shortest_path_to_point(
#                     env.controller,
#                     pos_unity,
#                     target
#                 )
#                 minimum_path_length = metrics.path_distance(path)
#
#                 rotation_allowed = False
#                 while not rotation_allowed:
#                     if len(possible_orientations) == 0:
#                         break
#                     rotation_y = random.choice(possible_orientations)
#                     possible_orientations.remove(rotation_y)
#                     evt = env.controller.step(
#                         action="TeleportFull",
#                         x=pos_unity['x'],
#                         y=pos_unity['y'],
#                         z=pos_unity['z'],
#                         rotation=dict(x=0, y=rotation_y, z=0)
#                     )
#                     rotation_allowed = evt.metadata['lastActionSuccess']
#                     if not evt.metadata['lastActionSuccess']:
#                         print(evt.metadata['errorMessage'])
#                         print("--------- Rotation not allowed! for pos {} rot {} ".format(pos_unity, rotation_y))
#
#                 if minimum_path_length > eps and rotation_allowed:
#                     scene_episodes.append({
#                         'scene': scene,
#                         'target_position': target,
#                         'initial_position': pos_unity,
#                         'initial_orientation': rotation_y,
#                         'shortest_path': path,
#                         'shortest_path_length': minimum_path_length
#                     })
#
#                 ninitials += 1
#             except ValueError:
#                 print("-----Invalid path discarding point...")
#
#         # scene=initials[:initials_per_target]
#         #
#         if ninitials > 0:
#             ntargets += 1
#         print('{} initial points accum for {} targets'.format(ninitials, ntargets))
#     print('{} targets in {}'.format(ntargets, scene))
#
#     sorted_objs = sorted(scene_episodes,
#                          key=lambda m: (m['shortest_path_length'], len(m['shortest_path'])))
#     third = int(len(sorted_objs) / 3.0)
#
#     for i, obj in enumerate(sorted_objs):
#         if i < third:
#             level = 'easy'
#         elif i < 2 * third:
#             level = 'medium'
#         else:
#             level = 'hard'
#         sorted_objs[i]['difficulty'] = level
#
#     print('{} easy {} medium {} hard in {}'.format(third, third, len(sorted_objs) - 2 * third, scene))
#
#     all_episodes += scene_episodes
#     print('{} episodes in {}'.format(len(scene_episodes), scene))
#
#     print('{} episodes in dataset'.format(len(all_episodes)))
#
#     fname = 'data/dataset_pointnav_{}.json'.format(scene)
#     with open(fname, 'w') as f:
#         json.dump(all_episodes, f, indent=4)
#
#     env.stop()
#
# @task
# def create_robothor_objectnav_dataset(ctx, rotateStepDegrees=45.0, visibilityDistance=1.0, gridSize=0.25, samples_per_scene=1000, width=300, height=300, fieldOfView=79):
#     from rl_robothor.robothor_environment import RoboThorEnvironment as Env
#     import random
#     import numpy as np
#     import copy
#     from ai2thor.util import metrics
#     import json
#
#     cfg = dict(
#         rotateStepDegrees=rotateStepDegrees,
#         visibilityDistance=visibilityDistance,
#         gridSize=gridSize,
#         width=width,
#         height=height,
#         fieldOfView=fieldOfView,
#         quality="Very Low",
#         continuousMode=True,
#         applyActionNoise=True,
#         agentType="stochastic",
#         snapToGrid=False,
#         agentMode="bot",
#         movementGaussianMu=1e-20,
#         movementGaussianSigma=1e-20,
#         rotateGaussianMu=1e-20,
#         rotateGaussianSigma=1e-20,
#         x_display="0.1"
#     )
#     env = Env(**cfg)
#
#     scenes = [
#         "FloorPlan_Train%d_%d" % (wall, furniture)
#         for wall in range(1, 13)
#         for furniture in range(1, 6)
#     ]
#
#     object_types = [
#         "AlarmClock",
#         "Apple",
#         "BaseballBat",
#         "BasketBall",
#         "Bowl",
#         "GarbageCan",
#         "HousePlant",
#         "Laptop",
#         "Mug",
#         "SprayBottle",
#         "Television",
#         "Vase",
#     ]
#
#     def extract_targets():
#         objs = env.all_objects()
#         targets = [obj for obj in objs if obj["objectType"] in object_types]
#         assert len(targets) == 12, "Found {} targets in scene {}".format(len(targets), env.scene_name)
#         return targets
#
#     def extract_initials(points, npoints, margin, tget):
#         initial = []
#         while len(initial) < npoints:
#             valid = False
#             nattempts = 0
#             while not valid and nattempts < 10:
#                 samp = random.choice(points)
#                 try:
#                     path = metrics.get_shortest_path_to_object_type(env.controller, tget["objectType"], samp)
#                     valid = True
#                     break
#                 except ValueError:
#                     points = [p for p in points if p != samp]
#                     continue
#                 nattempts += 1
#             if valid:
#                 initial.append(samp)
#                 points = [p for p in points if (p['x'] - samp['x']) ** 2 + (p['z'] - samp['z']) ** 2 > margin * margin]
#             else:
#                 print("Stopping after 10 attempts with {} initial points out of {} desired".format(len(points), npoints))
#                 break
#         return initial
#
#     all_episodes = []
#     for scene in scenes:
#         env.reset(scene)
#         candidates = copy.copy(env.currently_reachable_points)
#         targets = extract_targets()
#         # print(scene, len(targets), len(set([o['objectType'] for o in targets])))
#         scene_episodes = []
#         ntargets = 0
#         for target in targets:
#             initials = extract_initials(copy.copy(candidates), npoints=round(2 * samples_per_scene / len(targets)), margin=0.5 * gridSize, tget=target)
#             random.shuffle(initials)
#             initials = initials[:round(2 * samples_per_scene / len(targets))]
#
#             ninitials = 0
#             eps = 0.0001
#             for pos_unity in initials:
#                 possible_orientations = np.linspace(0, 360, num=round(360/rotateStepDegrees), endpoint=True).tolist()
#
#                 try:
#                     # print('{} to {} in {}'.format(pos_unity, target, scene))
#                     path = metrics.get_shortest_path_to_object_type(
#                         env.controller,
#                         target['objectType'],
#                         pos_unity,
#                     )
#                     minimum_path_length = metrics.path_distance(path)
#
#                     rotation_allowed = False
#                     while not rotation_allowed:
#                         if len(possible_orientations) == 0:
#                             break
#                         rotation_y = random.choice(possible_orientations)
#                         possible_orientations.remove(rotation_y)
#                         evt = env.controller.step(
#                             action="TeleportFull",
#                             x=pos_unity['x'],
#                             y=pos_unity['y'],
#                             z=pos_unity['z'],
#                             rotation=dict(x=0, y=rotation_y, z=0)
#                         )
#                         rotation_allowed = evt.metadata['lastActionSuccess']
#                         if not evt.metadata['lastActionSuccess']:
#                             print(evt.metadata['errorMessage'])
#                             print("--------- Rotation not allowed! for pos {} rot {} ".format(pos_unity, rotation_y))
#
#                     if minimum_path_length > eps and rotation_allowed:
#                         scene_episodes.append({
#                             'scene': scene,
#                             'target_position': target['position'],
#                             'object_type': target["objectType"],
#                             'object_id': target["objectId"],
#                             'initial_position': pos_unity,
#                             'initial_orientation': rotation_y,
#                             'shortest_path': path,
#                             'shortest_path_length': minimum_path_length
#                         })
#
#                     ninitials += 1
#                 except ValueError:
#                     print("-----Invalid path discarding point...")
#
#             # scene=initials[:initials_per_target]
#             #
#             if ninitials > 0:
#                 ntargets += 1
#         print('{} targets in {}'.format(ntargets, scene))
#
#         sorted_objs = sorted(scene_episodes,
#                              key=lambda m: (m['shortest_path_length'], len(m['shortest_path'])))
#         third = int(len(sorted_objs) / 3.0)
#
#         for i, obj in enumerate(sorted_objs):
#             if i < third:
#                 level = 'easy'
#             elif i < 2 * third:
#                 level = 'medium'
#             else:
#                 level = 'hard'
#             sorted_objs[i]['difficulty'] = level
#
#         print('{} easy {} medium {} hard in {}'.format(third, third, len(sorted_objs) - 2 * third, scene))
#
#         all_episodes += scene_episodes
#         print('{} episodes in {}'.format(len(scene_episodes), scene))
#
#     print('{} episodes in dataset'.format(len(all_episodes)))
#
#     fname = 'dataset_objectnav.json'
#     with open(fname, 'w') as f:
#         json.dump(all_episodes, f, indent=4)


@task
def dumm2(ctx):
    import random
    from rl_robothor.robothor_environment import RoboThorEnvironment as Env
    import cv2

    init_logging()

    e = Env(
        fieldOfView=59,#
        continuousMode=True,
        applyActionNoise=True,
        agentType="stochastic",
        rotateStepDegrees=45.0,
        visibilityDistance=1.0,
        gridSize=0.25,
        snapToGrid=False,
        agentMode="bot",
        width=400,
        height=300,
        movementGaussianMu=1e-20,
        movementGaussianSigma=1e-20,
        rotateGaussianMu=1e-20,
        rotateGaussianSigma=1e-20,
        include_private_scenes=True,
    )

    scenes = [
        # "FloorPlan_Train%d_%d" % (wall + 1, furniture + 1)
        # for wall in range(12)
        # for furniture in range(5)
        "FloorPlan_test-dev%d_%d" % (wall + 1, furniture + 1)
        for wall in range(2)
        for furniture in range(2)
    ]

    from typing import Any, Dict, Optional

    import gym
    import numpy as np
    import typing
    from rl_robothor.robothor_sensors import quaternion_rotate_vector, cartesian_to_polar, quaternion_from_y_angle

    from rl_robothor.robothor_environment import RoboThorEnvironment
    from rl_base.sensor import Sensor

    class LeTask(dict):
        def __init__(self, tget):
            super().__init__()
            self["target"] = tget

    class TargetCoordinatesSensorRobothor(Sensor[RoboThorEnvironment, LeTask]):
        def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
            super().__init__(config, *args, **kwargs)

            self.observation_space = gym.spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )

        def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
            return "target_coordinates_ind"

        def _get_observation_space(self) -> gym.spaces.Box:
            return typing.cast(gym.spaces.Box, self.observation_space)

        def _compute_pointgoal(self, source_position, source_rotation, goal_position):
            direction_vector = goal_position - source_position
            direction_vector_agent = quaternion_rotate_vector(
                source_rotation.inverse(), direction_vector
            )
            print("sr", source_rotation)
            print("dv", direction_vector)
            print("dvagent", direction_vector_agent)

            rho, phi = cartesian_to_polar(
                direction_vector_agent[2], -direction_vector_agent[0]
            )
            return np.array([rho, phi], dtype=np.float32)

        def get_observation(
                self, env: RoboThorEnvironment, task, *args: Any, **kwargs: Any
        ) -> Any:
            agent_state = env.agent_state()
            agent_position = np.array([agent_state[k] for k in ["x", "y", "z"]])
            rotation_world_agent = quaternion_from_y_angle(agent_state["rotation"]["y"])

            goal_position = np.array([task["target"][k] for k in ["x", "y", "z"]])

            return self._compute_pointgoal(
                agent_position, rotation_world_agent, goal_position
            )

    def target_reset():
        scene = random.choice(scenes)
        e.reset(scene)
        print(scene)

        e.randomize_agent_location()
        print(e.last_action_success)

        target = random.choice(e.currently_reachable_points)
        print(len(e.currently_reachable_points), "reachable points")

        return LeTask(target), e.agent_state()

    compass = TargetCoordinatesSensorRobothor({})
    task, reset_state = target_reset()
    steps = 0
    while True:
        print(
            "{} {} {} {}".format(
                steps, e.agent_state(), task, compass.get_observation(e, task)
            )
        )
        cv2.imshow("dumm2", e.current_frame[:, :, ::-1])
        key = cv2.waitKey(0)
        if key == ord("w"):
            e.step({"action": "MoveAhead"})
            print(e.last_action_success)
            if not e.last_action_success:
                print(e.last_event.metadata["errorMessage"])
            steps += 1
        elif key == ord("a"):
            e.step({"action": "RotateLeft"})
            print(e.last_action_success)
            steps += 1
        elif key == ord("d"):
            e.step({"action": "RotateRight"})
            print(e.last_action_success)
            steps += 1
        elif key == ord("s"):
            e.step({"action": "MoveBack"})
            print(e.last_action_success)
            steps += 1
        elif key == ord("z"):
            e.step({"action": "LookUp"})
            print(e.last_action_success)
            steps += 1
        elif key == ord("x"):
            e.step({"action": "LookDown"})
            print(e.last_action_success)
            steps += 1
        elif key == ord("r"):
            # e.reset()
            e.step({"action": "TeleportFull", **reset_state})
            print(e.last_action_success)
            steps = 0
        elif key == ord("t"):
            target, reset_state = target_reset()
            steps = 0
        elif key == ord("q"):
            break

    e.stop()

# @task
# def top_down_views(ctx):
#     for scene in
#
# @task
# def convert_val_points(ctx):
#     import json
#     with open()

@task
def time_detector(ctx):
    import time
    import torch
    from utils.cacheless_frcnn import fasterrcnn_resnet50_fpn
    from torch.nn.parallel.distributed import DistributedDataParallel
    from rl_robothor.robothor_preprocessors import BatchedFasterRCNN
    import cv2
    import random

    ndevices = torch.cuda.device_count()
    nimages = max(20 * ndevices, 1)
    npasses = 3

    imwidth = 400  # 640
    imheight = 300  # 480

    print("Loading model")
    model = BatchedFasterRCNN().to(torch.device("cuda:0") if ndevices > 0 else torch.device("cpu"))

    print("Distributing model")
    if ndevices > 0:
        # # DistributedDataParallel
        store = torch.distributed.TCPStore("localhost", 4712, 1, True)
        torch.distributed.init_process_group(backend="nccl", store=store, rank=0, world_size=1)
        model = DistributedDataParallel(model, device_ids=list(range(ndevices)))

        # # DataParallel
        # model = torch.nn.DataParallel(model, device_ids=list(range(ndevices)))

    print("Creating images")
    images = torch.cat([(1. / (1. + it)) * torch.ones(1, 3, imheight, imwidth) for it in range(nimages)], dim=0)
    # images = torch.cat([torch.from_numpy(cv2.imread('test{}.png'.format(random.randint(0, 3)))[:, :, ::-1] / 255.0).to(torch.float32).permute(2, 0, 1).unsqueeze(0) for _ in range(nimages)], dim=0)

    for timages in [nimages, 120, 112, 104, 100, 96, 88, 80, 72, 64, 60, 56, 50, 48, 40, 32, 30, 24, 20, 16, 10, ndevices]:
        if timages > images.shape[0] or timages == 0:
            continue

        # warm up
        with torch.no_grad():
            classes, boxes = model(images[:timages, ...])
            # print('classes', classes.shape, 'boxes', boxes.shape)
            del classes, boxes

        # measure
        ttime = -time.time()
        for it in range(npasses):
            with torch.no_grad():
                _ = model(images[:timages, ...])
        ttime += time.time()

        # for device in range(ndevices):
        #     print('{0} reserved {1:.2f} GB allocated {2:.2f} MB'.format(
        #         device,
        #         torch.cuda.memory_reserved(device) / (1024 * 1024 * 1024),
        #         torch.cuda.memory_allocated(device) / (1024 * 1024)
        #     ))

        print("{0} samples {1:.2f} fps".format(timages, (npasses * timages) / ttime))


def my_worker(id, parent, seed=None):
    import time
    import random
    import torch
    import numpy as np

    init_logging()
    set_seed(seed)

    LOGGER.info("Parent {} Worker {} torch {} numpy {} random {}".format(parent, id, torch.initial_seed(), np.random.rand(), random.random()))
    # time.sleep(random.random() * 2)
    # LOGGER.info("Parent {} Worker {}".format(parent, id))


def my_trainer(id, world_size, nworkers, mp_ctx, port, model_seed, process_seed=None, barrier=None):
    import torch
    import random
    import numpy as np

    import signal
    import sys

    def sigterm_handler(_signo, _stack_frame):
        # Raises SystemExit(0):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, sigterm_handler)

    init_logging()

    LOGGER.debug("Trainer {}".format(id))
    LOGGER.info("Trainer {}".format(id))
    LOGGER.warning("Trainer {}".format(id))
    LOGGER.error("Trainer {}".format(id))

    LOGGER.info("Trainer {} {} waiting".format(id, barrier.n_waiting))
    id = barrier.wait()
    if id == 0:
        barrier.reset()
        LOGGER.info("Trainer {} resets barrier".format(id))
        LOGGER.info("Trainer {} {} waiting".format(id, barrier.n_waiting))

    LOGGER.info("Trainer {} after barrier".format(id))

    import time
    try:
        LOGGER.info("Trainer {} sleeping".format(id))
        time.sleep(100)
    except KeyboardInterrupt:
        LOGGER.info("Leaving after keyboard interrupt")
    except Exception:
        LOGGER.info("Exception")
    finally:
        LOGGER.info("Returning")
        return

    assert 2 == 1, "enforcing failure in worker {}".format(id)

    store = torch.distributed.TCPStore("localhost", port, world_size, id == 0)
    torch.distributed.init_process_group(backend="nccl", store=store, rank=id, world_size=world_size)

    LOGGER.info("Trainer {} logged in".format(id))

    set_seed(model_seed)
    device = torch.device("cuda:{}".format(id % torch.cuda.device_count()))
    model = torch.nn.Linear(3, 1, bias=False).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, [device], device)

    set_seed(process_seed)
    worker_seeds = [random.randint(0, 65535) for _ in range(nworkers)]

    LOGGER.info("Trainer {} created {} with seed {} and set seed to {}".format(id, model.module.state_dict(), model_seed, process_seed))
    LOGGER.info("Parent {} torch {} numpy {} random {}".format(id, torch.initial_seed(), np.random.rand(), random.random()))
    LOGGER.info("Parent {} worker seeds {}".format(id, worker_seeds))

    ps = []
    for wid in range(nworkers):
        ps.append(mp_ctx.Process(target=my_worker, args=(wid, id, worker_seeds[wid])))
        ps[-1].daemon = True
        ps[-1].start()

    for p in ps:
        p.join()

    LOGGER.info("Trainer {} done".format(id))


@task
def inherit_tcpstore(ctx, run_seed=12345):
    import torch
    import torch.multiprocessing as mp
    import random
    import time

    init_logging()

    # assert 2 == 1, "oops"

    port = 4712
    # world_size = torch.cuda.device_count()
    world_size = 3
    nworkers = 3

    set_seed(run_seed)
    model_seed = random.randint(0, 65535)
    worker_seeds = [random.randint(0, 65535) for _ in range(world_size)]

    mp_ctx = mp.get_context("fork")
    # mp_ctx = mp.get_context("forkserver")
    barrier = mp.Barrier(world_size)
    ps = []
    for id in range(world_size):
        ps.append(mp_ctx.Process(target=my_trainer, args=(id, world_size, nworkers, mp_ctx, port, model_seed, worker_seeds[id], barrier)))
        ps[-1].daemon = False
        ps[-1].start()

    time.sleep(5)

    for p in ps:
        LOGGER.info('Process alive {}'.format(p.is_alive()))
        if p.is_alive():
            p.terminate()
            p.join()
        else:
            p.join()

    LOGGER.info("Done")
