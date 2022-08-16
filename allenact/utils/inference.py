from typing import Optional, cast, Tuple, Any, Dict

import attr
import torch

from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
from allenact.algorithms.onpolicy_sync.storage import RolloutStorage
from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
from allenact.base_abstractions.misc import (
    Memory,
    ObservationType,
    ActorCriticOutput,
    DistributionType,
)
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.utils import spaces_utils as su
from allenact.utils.tensor_utils import batch_observations


@attr.s(kw_only=True)
class InferenceAgent:
    actor_critic: ActorCriticModel = attr.ib()
    rollout_storage: RolloutStorage = attr.ib()
    device: torch.device = attr.ib()
    sensor_preprocessor_graph: Optional[SensorPreprocessorGraph] = attr.ib()
    steps_before_rollout_refresh: int = attr.ib(default=128)
    memory: Optional[Memory] = attr.ib(default=None)
    steps_taken_in_task: int = attr.ib(default=0)
    last_action_flat: Optional = attr.ib(default=None)
    has_initialized: Optional = attr.ib(default=False)

    def __attrs_post_init__(self):
        self.actor_critic.eval()
        self.actor_critic.to(device=self.device)
        if self.memory is not None:
            self.memory.to(device=self.device)
        if self.sensor_preprocessor_graph is not None:
            self.sensor_preprocessor_graph.to(self.device)

        self.rollout_storage.to(self.device)
        self.rollout_storage.set_partition(index=0, num_parts=1)

    @classmethod
    def from_experiment_config(
        cls,
        exp_config: ExperimentConfig,
        device: torch.device,
        checkpoint_path: Optional[str] = None,
        model_state_dict: Optional[Dict[str, Any]] = None,
        mode: str = "test",
    ):
        assert (
            checkpoint_path is None or model_state_dict is None
        ), "Cannot have `checkpoint_path` and `model_state_dict` both non-None."
        rollout_storage = exp_config.training_pipeline().rollout_storage

        machine_params = exp_config.machine_params(mode)
        if not isinstance(machine_params, MachineParams):
            machine_params = MachineParams(**machine_params)

        sensor_preprocessor_graph = machine_params.sensor_preprocessor_graph

        actor_critic = cast(
            ActorCriticModel,
            exp_config.create_model(
                sensor_preprocessor_graph=sensor_preprocessor_graph
            ),
        )

        if checkpoint_path is not None:
            actor_critic.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
            )
        elif model_state_dict is not None:
            actor_critic.load_state_dict(
                model_state_dict
                if "model_state_dict" not in model_state_dict
                else model_state_dict["model_state_dict"]
            )

        return cls(
            actor_critic=actor_critic,
            rollout_storage=rollout_storage,
            device=device,
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    def reset(self):
        if self.has_initialized:
            self.rollout_storage.after_updates()
        self.steps_taken_in_task = 0
        self.memory = None

    def act(self, observations: ObservationType):
        # Batch of size 1
        obs_batch = batch_observations([observations], device=self.device)
        if self.sensor_preprocessor_graph is not None:
            obs_batch = self.sensor_preprocessor_graph.get_observations(obs_batch)

        if self.steps_taken_in_task == 0:
            self.has_initialized = True
            self.rollout_storage.initialize(
                observations=obs_batch,
                num_samplers=1,
                recurrent_memory_specification=self.actor_critic.recurrent_memory_specification,
                action_space=self.actor_critic.action_space,
            )
            self.rollout_storage.after_updates()
        else:
            dummy_val = torch.zeros((1, 1), device=self.device)  # Unused dummy value
            self.rollout_storage.add(
                observations=obs_batch,
                memory=self.memory,
                actions=self.last_action_flat[0],
                action_log_probs=dummy_val,
                value_preds=dummy_val,
                rewards=dummy_val,
                masks=torch.ones(
                    (1, 1), device=self.device
                ),  # Always == 1 as we're in a single task until `reset`
            )

        agent_input = self.rollout_storage.agent_input_for_next_step()

        actor_critic_output, self.memory = cast(
            Tuple[ActorCriticOutput[DistributionType], Optional[Memory]],
            self.actor_critic(**agent_input),
        )

        action = actor_critic_output.distributions.sample()
        self.last_action_flat = su.flatten(self.actor_critic.action_space, action)

        self.steps_taken_in_task += 1

        if self.steps_taken_in_task % self.steps_before_rollout_refresh == 0:
            self.rollout_storage.after_updates()

        return su.action_list(self.actor_critic.action_space, self.last_action_flat)[0]
