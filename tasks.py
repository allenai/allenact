from invoke import task


# No path for FloorPlan_Train6_4 from
# {'x': 6.0, 'y': 0.9009997, 'z': -3.6035533, 'rotation': {'x': 0.0, 'y': 179.999985, 'z': 0.0}, 'horizon': 0.0}
# to {'x': 5.25, 'y': 0.9009997, 'z': -4.25}

@task
def reachable_pos(ctx, scene="FloorPlan_Train2_3", editor_mode=False, local_build=False):
    import ai2thor.util.metrics as metrics
    import ai2thor.controller
    from ai2thor.util.metrics import get_shortest_path_to_point, get_shortest_path_to_object_type
    import copy
    gridSize = 0.25

    controller = ai2thor.controller.Controller(
        # rotateStepDegrees=45,
        visibilityDistance=1.0,
        gridSize=gridSize,
        # port=8200,
        # host='127.0.0.1',
        # local_executable_path=_local_build_path() if local_build else None,
        # start_unity=False if editor_mode else True,
        agentType="stochastic",
        continuousMode=True,
        # continuous=False,
        snapToGrid=False,
        agentMode="bot",
        scene=scene,
        width=300,
        height=300,

        fieldOfView=79.0,
        applyActionNoise=True,
        rotateStepDegrees=45.0,
        movementGaussianMu=1e-20,  # almost deterministic
        movementGaussianSigma=1e-20,  # almost deterministic
        rotateGaussianMu=1e-20,  # almost deterministic
        rotateGaussianSigma=1e-20,  # almost deterministic

        # continus=True
    )
    print("constoller.last_action Agent Pos: {}".format(controller.last_event.metadata["agent"]["position"]))

    evt = controller.step(action="GetReachablePositions", gridSize=gridSize)
    print("After GetReachable AgentPos: {}".format(evt.metadata['agent']['position']))
    print(evt.metadata["lastActionSuccess"])
    # print(evt.metadata["errorMessage"])
    reachable_pos = evt.metadata["actionReturn"]
    # print(evt.metadata["actionReturn"])

    evt = controller.step(
        dict(
            action="TeleportFull",
            x=8.0,
            y=reachable_pos[0]['y'],
            z=-2.25,
            rotation=dict(x=0, y=180.0, z=0),
            horizon=0.0,
        )
    )
    print("After teleport: {} {} {}".format(evt.metadata['agent']['position'], evt.metadata['agent']['rotation']['y'], evt.metadata['agent']['cameraHorizon']))
    source = copy.deepcopy(evt.metadata['agent']['position'])
    # source['y'] += 0.02
    print(get_shortest_path_to_point(controller, source, {'x': 9.0, 'y': 0.900999665, 'z': -3.75}))
    print("After path to point: {} {} {}".format(evt.metadata['agent']['position'], evt.metadata['agent']['rotation']['y'], evt.metadata['agent']['cameraHorizon']))

    evt = controller.step(
            action="TeleportFull",
            x=8.0,
            y=reachable_pos[0]['y'],
            z=-2.25,
            rotation=dict(x=0, y=180.0, z=0),
            horizon=0.0,
    )
    print(controller.last_event.metadata["lastActionSuccess"])
    print("After re-teleport: {} {} {}".format(evt.metadata['agent']['position'], evt.metadata['agent']['rotation']['y'], evt.metadata['agent']['cameraHorizon']))

    get_shortest_path_to_object_type(controller, "Television", source)  #, copy.deepcopy(evt.metadata['agent']['rotation']))
    evt = controller.last_event
    print("After path to object: {} {} {}".format(evt.metadata['agent']['position'], evt.metadata['agent']['rotation']['y'], evt.metadata['agent']['cameraHorizon']))

    evt = controller.step(
            action="TeleportFull",
            x=8.0,
            y=reachable_pos[0]['y'],
            z=-2.25,
            rotation=dict(x=0, y=180.0, z=0),
            horizon=0.0,
    )
    print(controller.last_event.metadata["lastActionSuccess"])
    print("After re-teleport: {} {} {}".format(evt.metadata['agent']['position'], evt.metadata['agent']['rotation']['y'], evt.metadata['agent']['cameraHorizon']))

    controller.stop()





    # TARGET_TYPES = sorted(
    #     [
    #         "AlarmClock",
    #         "Apple",
    #         "BaseballBat",
    #         "BasketBall",
    #         "Bowl",
    #         "GarbageCan",
    #         "HousePlant",
    #         "Laptop",
    #         "Mug",
    #         "Remote",
    #         "SprayBottle",
    #         "Television",
    #         "Vase",
    #         # 'AlarmClock',
    #         # 'Apple',
    #         # 'BasketBall',
    #         # 'Mug',
    #         # 'Television',
    #     ]
    # )
    #
    # TRAIN_SCENES = [
    #     "FloorPlan_Train%d_%d" % (wall + 1, furniture + 1)
    #     for wall in range(12)
    #     for furniture in range(5)
    # ]
    #
    # VALID_SCENES = [
    #     "FloorPlan_Val%d_%d" % (wall + 1, furniture + 1)
    #     for wall in range(3)
    #     for furniture in range(5)
    # ]
    #
    # env_config = dict(
    #     visibilityDistance=1.0,
    #     gridSize=gridSize,
    #     agentType="stochastic",
    #     continuousMode=True,
    #     snapToGrid=False,
    #     agentMode="bot",
    #     scene=scene,
    #     width=300,
    #     height=300,
    #     fieldOfView=79.0,
    #     applyActionNoise=True,
    #     rotateStepDegrees=45.0,
    #     movementGaussianMu=1e-20,  # almost deterministic
    #     movementGaussianSigma=1e-20,  # almost deterministic
    #     rotateGaussianMu=1e-20,  # almost deterministic
    #     rotateGaussianSigma=1e-20,  # almost deterministic
    # )
    #
    # from rl_robothor.robothor_task_samplers import ObjectNavTaskSampler, ObjectNavTask, PointNavTaskSampler, PointNavTask
    # from rl_ai2thor.ai2thor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
    # import gym
    # from utils.system import LOGGER, init_logging
    # init_logging()
    #
    # SCREEN_SIZE = 224
    #
    # SENSORS = [
    #     RGBSensorThor(
    #         {
    #             "height": SCREEN_SIZE,
    #             "width": SCREEN_SIZE,
    #             "use_resnet_normalization": True,
    #             "uuid": "rgb_lowres",
    #         }
    #     ),
    #     GoalObjectTypeThorSensor({
    #         "object_types": TARGET_TYPES,
    #     }),
    # ]
    #
    # sampler_args = {
    #     "scenes": TRAIN_SCENES,
    #     "object_types": TARGET_TYPES,
    #     "sensors": SENSORS,
    #     "max_steps": 10,
    #     "action_space": gym.spaces.Discrete(len(ObjectNavTask.action_names())),
    #     "rewards_config": {
    #         "step_penalty": -0.01,
    #         "goal_success_reward": 10.0,
    #         "failed_stop_reward": 0.0,
    #         "shaping_weight": 1.0,  # applied to the decrease in distance to target
    #     },
    #     "env_args": env_config,
    # }
    #
    # sampler = ObjectNavTaskSampler(**sampler_args)
    # while True:
    #     task = sampler.next_task()
    #     # LOGGER.info("{}".format(task.task_info))
    #     # LOGGER.info("{}".format(task.env.dist_to_object(task.task_info['object_type'])))
