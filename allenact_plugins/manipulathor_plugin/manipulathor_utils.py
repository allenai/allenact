import ai2thor

from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment


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
    return event
