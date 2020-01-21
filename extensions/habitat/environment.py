"""A wrapper for interacting with the Habitat environment"""

import habitat

class HabitatEnvironment(object):
    def __init__(
            self,
            docker_enabled: bool = False,
            x_display: str = None
                ) -> None:
        print("habitat env constructor")
        self.x_display = x_display
        self.controller = habitat.Env (
            config=habitat.get_config("configs/tasks/pointnav.yaml")
        )
