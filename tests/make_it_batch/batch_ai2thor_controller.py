from typing import List
from ai2thor.controller import Controller
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment


class BatchController:
    def __init__(
            self,
            task_batch_size: int,
            **kwargs,
    ):
        self.task_batch_size = task_batch_size
        self.controllers = [IThorEnvironment(**kwargs) for _ in range(task_batch_size)]
        self._frames = []

    def step(self, actions: List[str]):
        assert len(actions) == self.task_batch_size
        for controller, action in zip(self.controllers, actions):
            controller.step(action=action if action != "End" else "Pass")
        self._frames = []
        return self.batch_last_event()

    def get_agent_location(self):
        return None

    def reset(
            self,
            idx: int,
            scene: str,
    ):
        self.controllers[idx].reset(scene)

    def batch_reset(
            self,
            scenes: List[str],
    ):
        for controller, scene in zip(self.controllers, scenes):
            controller.reset(scene)

    def stop(self):
        for controller in self.controllers:
            controller.stop()

    def last_event(self, idx: int):
        return self.controllers[idx].last_event

    def batch_last_event(self):
        return [controller.last_event for controller in self.controllers]

    def render(self):
        assert len(self._frames) == 0
        for controller in self.controllers:
            self._frames.append(controller.last_event.frame)
