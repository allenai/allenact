import ai2thor

from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.manipulathor_plugin.armpointnav_constants import (
    get_agent_start_positions,
)
from allenact_plugins.manipulathor_plugin.manipulathor_constants import (
    ADDITIONAL_ARM_ARGS,
)


def make_all_objects_unbreakable(controller):
    all_breakable_objects = [
        o["objectType"]
        for o in controller.last_event.metadata["objects"]
        if o["breakable"] is True
    ]
    all_breakable_objects = set(all_breakable_objects)
    for obj_type in all_breakable_objects:
        controller.step(action="MakeObjectsOfTypeUnbreakable", objectType=obj_type)


def reset_environment_and_additional_commands(controller, scene_name):
    controller.reset(scene_name)
    controller.step(action="MakeAllObjectsMoveable")
    controller.step(action="MakeObjectsStaticKinematicMassThreshold")
    make_all_objects_unbreakable(controller)
    return


def transport_wrapper(controller, target_object, target_location):
    transport_detail = dict(
        action="PlaceObjectAtPoint",
        objectId=target_object,
        position=target_location,
        forceKinematic=True,
    )
    advance_detail = dict(action="AdvancePhysicsStep", simSeconds=1.0)

    if issubclass(type(controller), IThorEnvironment):
        event = controller.step(transport_detail)
        controller.step(advance_detail)
    elif type(controller) == ai2thor.controller.Controller:
        event = controller.step(**transport_detail)
        controller.step(**advance_detail)
    else:
        raise NotImplementedError
    return event


def initialize_arm(controller):
    # for start arm from high up,
    scene = controller.last_event.metadata["sceneName"]
    initial_pose = get_agent_start_positions()[scene]
    event1 = controller.step(
        dict(
            action="TeleportFull",
            standing=True,
            x=initial_pose["x"],
            y=initial_pose["y"],
            z=initial_pose["z"],
            rotation=dict(x=0, y=initial_pose["rotation"], z=0),
            horizon=initial_pose["horizon"],
        )
    )
    event2 = controller.step(
        dict(action="MoveArm", position=dict(x=0.0, y=0, z=0.35), **ADDITIONAL_ARM_ARGS)
    )
    event3 = controller.step(dict(action="MoveArmBase", y=0.8, **ADDITIONAL_ARM_ARGS))
    return event1, event2, event3
