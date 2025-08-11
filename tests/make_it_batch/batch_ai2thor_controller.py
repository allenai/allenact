from typing import List
from ai2thor.controller import Controller


class BatchController:
    def __init__(
        self, task_batch_size: int, **kwargs,
    ):
        self.task_batch_size = task_batch_size
        self.controllers = [Controller(**kwargs) for _ in range(task_batch_size)]

    def step(self, actions: List[str]):
        assert len(actions) == self.task_batch_size
        for controller, action in zip(self.controllers, actions):
            controller.step(action)
        return self.batch_last_event()

    def reset(
        self, idx: int, scene: str,
    ):
        self.controllers[idx].reset(scene)

    def batch_reset(
        self, scenes: List[str],
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
        frames = []
        for controller in self.controllers:
            frames.append(controller.last_event.frame)
        return frames
