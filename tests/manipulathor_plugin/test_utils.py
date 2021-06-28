from allenact_plugins.manipulathor_plugin.arm_calculation_utils import world_coords_to_agent_coords


class TestArmCalculationUtils(object):

    def test_translation_functions(self):
        agent_coordinate = {
            "position": {"x": 1, "y": 0, "z": 2},
            "rotation": {"x": 0, "y": -45, "z": 0},
        }
        obj_coordinate = {
            "position": {"x": 0, "y": 1, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0},
        }
        rotated = world_coords_to_agent_coords(obj_coordinate, agent_coordinate)
        eps = 0.01
        assert (
                rotated["position"]["x"] - (-2.1) < eps
                and rotated["position"]["x"] - (1) < eps
                and rotated["position"]["x"] - (-0.7) < eps
        )


if __name__ == "__main__":
    TestArmCalculationUtils().test_translation_functions()