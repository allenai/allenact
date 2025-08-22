from collections import Counter

import torch

from allenact.utils.experiment_utils import set_seed
from allenact.utils.inference import InferenceAgent
from projects.babyai_baselines.experiments.go_to_obj.ppo import (
    PPOBabyAIGoToObjExperimentConfig,
)

from packaging.version import parse

if parse(torch.__version__) >= parse("2.0.0"):
    expected_results = [
        {
            "ep_length": 39,
            "reward": 0.45999999999999996,
            "task_info": {},
            "success": 1.0,
        },
        {"ep_length": 64, "reward": 0.0, "task_info": {}, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "task_info": {}, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "task_info": {}, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "task_info": {}, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "task_info": {}, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "task_info": {}, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "task_info": {}, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "task_info": {}, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "task_info": {}, "success": 0.0},
    ]
else:
    expected_results = [
        {"ep_length": 64, "reward": 0.0, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "success": 0.0},
        {"ep_length": 17, "reward": 0.7646153846153846, "success": 1.0},
        {"ep_length": 22, "reward": 0.6953846153846154, "success": 1.0},
        {"ep_length": 64, "reward": 0.0, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "success": 0.0},
        {"ep_length": 64, "reward": 0.0, "success": 0.0},
    ]


class TestInferenceAgent(object):
    def test_inference_agent_from_minigrid_config(self):
        set_seed(1)

        exp_config = PPOBabyAIGoToObjExperimentConfig()
        agent = InferenceAgent.from_experiment_config(
            exp_config=exp_config,
            device=torch.device("cpu"),
        )

        task_sampler = exp_config.make_sampler_fn(
            **exp_config.test_task_sampler_args(process_ind=0, total_processes=1)
        )

        all_actions = []
        successes = 0
        for ind, expected_result in zip(range(10), expected_results):
            agent.reset()

            task = task_sampler.next_task()
            observations = task.get_observations()

            actions = []
            while not task.is_done():
                action = agent.act(observations=observations)
                actions.append(action)
                observations = task.step(action).observation

            metrics = task.metrics()

            successes += metrics["success"]

            assert metrics["success"] == 0 or metrics["reward"] > 0
            assert metrics["ep_length"] <= 64

            # Random seeding seems to not work well when changing linux/mac and torch versions :(
            # assert all(
            #     abs(v - expected_result[k]) < 1e-4
            #     for k, v in task.metrics().items()
            #     if k != "task_info"
            # ), f"Failed on task {ind} with actions {actions} and metrics {task.metrics()} (expected={expected_result})."

            all_actions.append(actions)

        assert successes > 0, "At least one task should be successful hopefully..."
        assert min(Counter(sum(all_actions, [])).values()) >= len(
            sum(all_actions, [])
        ) * 1 / (7 + 3), (
            "Statistically, all actions should be taken at around 1/7 * num_actions times. We add 3 to"
            " the denominator for unlikely settings."
        )


if __name__ == "__main__":
    TestInferenceAgent().test_inference_agent_from_minigrid_config()
