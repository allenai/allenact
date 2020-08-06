import math
import os

import py

from constants import ABS_PATH_OF_TOP_LEVEL_DIR
from common.algorithms.onpolicy_sync.runner import OnPolicyRunner
from projects.babyai_baselines.experiments.go_to_obj.ppo import (
    PPOBabyAIGoToObjExperimentConfig,
)


class TestGoToObjTrains(object):
    def test_ppo_trains(
        self, tmpdir=py.path.local(os.path.join(ABS_PATH_OF_TOP_LEVEL_DIR, "tests/tmp"))
    ):
        cfg = PPOBabyAIGoToObjExperimentConfig()

        tmpdir = tmpdir.join("experiment_output")
        if not tmpdir.exists():
            tmpdir.mkdir()
        output_dir: str = tmpdir.dirname

        train_runner = OnPolicyRunner(
            config=cfg,
            output_dir=output_dir,
            loaded_config_src_files=None,
            seed=1,
            mode="train",
            deterministic_cudnn=True,
        )

        start_time_str = train_runner.start_train(max_sampler_processes_per_worker=1)

        test_runner = OnPolicyRunner(
            config=cfg,
            output_dir=output_dir,
            loaded_config_src_files=None,
            seed=1,
            mode="test",
            deterministic_cudnn=True,
        )
        test_results = test_runner.start_test(
            experiment_date=start_time_str,
            skip_checkpoints=1,
            max_sampler_processes_per_worker=1,
        )

        assert len(test_results) == 1, "Too many test results"

        tr = test_results[0]
        assert (
            tr["training_steps"]
            == round(
                math.ceil(
                    cfg.TOTAL_RL_TRAIN_STEPS
                    / (cfg.ROLLOUT_STEPS * cfg.NUM_TRAIN_SAMPLERS)
                )
            )
            * cfg.ROLLOUT_STEPS
            * cfg.NUM_TRAIN_SAMPLERS
        ), "Incorrect number of training steps"
        assert len(tr["tasks"]) == cfg.NUM_TEST_TASKS, "Incorrect number of test tasks"
        assert tr["success"] == sum(task["success"] for task in tr["tasks"]) / len(
            tr["tasks"]
        ), "Success counts don't seem to match"
        assert (
            tr["success"] > 0.95
        ), "PPO did not seem to converge for the go_to_obj task."


if __name__ == "__main__":
    TestGoToObjTrains().test_ppo_trains()
