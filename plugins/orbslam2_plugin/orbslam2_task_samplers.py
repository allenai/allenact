from typing import Optional

from core.base_abstractions.task import TaskSampler
from plugins.robothor_plugin.robothor_tasks import PointNavTask
from plugins.robothor_plugin import robothor_task_samplers


class PointNavDatasetTaskSampler(robothor_task_samplers.PointNavDatasetTaskSampler):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.orbslam_sensor = next(s for s in self.sensors if s.uuid == kwargs['orbslam_uuid'])

    def next_task(self, force_advance_scene: bool = False) -> Optional[PointNavTask]:
        task = super().next_task(force_advance_scene)
        if task is not None:
            self.orbslam_sensor.reset(self.env.agent_state())
        return task
