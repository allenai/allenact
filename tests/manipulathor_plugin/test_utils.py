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
                abs(rotated["position"]["x"] - (-2.12)) < eps
                and abs(rotated["position"]["y"] - (1.0)) < eps
                and abs(rotated["position"]["z"] - (-0.70)) < eps
        )


if __name__ == "__main__":
    TestArmCalculationUtils().test_translation_functions()