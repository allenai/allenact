import math

from core.algorithms.onpolicy_sync.runner import OnPolicyRunner
from projects.babyai_baselines.experiments.go_to_obj.ppo import (
    PPOBabyAIGoToObjExperimentConfig,
)


class TestGoToObjTrains(object):
    def test_ppo_trains(self, tmpdir):
        cfg = PPOBabyAIGoToObjExperimentConfig()

        output_dir = tmpdir.mkdir("experiment_output")

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
        ), "PPO did not seem to converge for the go_to_obj task (success {}).".format(
            tr["success"]
        )


if __name__ == "__main__":
    import pathlib

    TestGoToObjTrains().test_ppo_trains(pathlib.Path("testing"))  # type:ignore
