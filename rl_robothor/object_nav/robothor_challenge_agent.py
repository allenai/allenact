import torch

# from robothor_challenge.robothor_challenge.agent import Agent

from rl_base.sensor import SensorSuite
from onpolicy_sync.storage import RolloutStorage
from utils.tensor_utils import batch_observations

from rl_robothor.object_nav.tasks import ObjectNavTask as TrainingTask
from experiments.robothor.object_nav_resnet_tensor_50train_5tget import (
    ObjectNavRoboThorExperimentConfig as ExperimentConfig,
)

checkpoint_name = "/Users/jordis/Desktop/exp_object_nav_resnet_tensor_50train_5tget__time_2020-02-13_08-48-11__stage_00__steps_000021607500__seed_936652430.pt"

from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, episode):
        self.episode = episode

    @abstractmethod
    def on_event(self, event):
        pass


class ResnetTensorObjNavAgent(Agent):
    def __init__(self, episode):
        super().__init__(episode)
        self.engine = SingletonEngine.instance()
        self.engine.reset(self.episode)

    def on_event(self, event):
        return self.engine.step(event)


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError("Singletons must be accessed through `instance()`.")

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


class State:
    action = None
    event = None
    actor_critic_output = None
    recurrent_hidden_states = None
    rollouts = None


@Singleton
class SingletonEngine:
    def __init__(self):
        self._actions = list(TrainingTask.action_names())

        self.config = ExperimentConfig()
        self.device = self.pick_device()
        self.sensors = self.sensor_suite = SensorSuite(self.config.SENSORS)

        self.machine_params = self.config.machine_params("test")
        if "observation_set" in self.machine_params:
            self.observation_set = self.machine_params["observation_set"].to(
                self.device
            )
            self.actor_critic = self.config.create_model(
                observation_set=self.observation_set
            ).to(self.device)
        else:
            self.observation_set = None
            self.actor_critic = self.config.create_model().to(self.device)

        self.checkpoint_load(checkpoint_name)

        self.task_info = None
        self.state = None

    def pick_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)  # type: ignore
        else:
            device = torch.device("cpu")
        return device

    def checkpoint_load(self, ckpt):
        ckpt = torch.load(ckpt, map_location="cpu")
        self.actor_critic.load_state_dict(ckpt["model_state_dict"])
        self.actor_critic.eval()

    def reset(self, episode):
        self.task_info = episode  # provides Task interface to sensors
        self.state = State()
        self.state.rollouts = RolloutStorage(
            num_steps=1,
            num_processes=1,
            action_space=self.actor_critic.action_space,
            recurrent_hidden_state_size=self.actor_critic.recurrent_hidden_state_size,
            num_recurrent_layers=self.actor_critic.num_recurrent_layers,
        )
        self.state.rollouts.to(self.device)

    def preprocess(self, batched_observations):
        if self.observation_set is None:
            return batched_observations
        return self.observation_set.get_observations(batched_observations)

    @property
    def current_frame(self):  # provides Environment interface to sensors
        return self.state.event.frame

    def update_state(self, event):
        self.state.event = event  # provides Environment interface to sensors

        obs = self.preprocess(
            batch_observations(
                [self.sensor_suite.get_observations(self, self)], self.device
            )
        )

        if self.state.action is None:
            self.state.rollouts.insert_initial_observations(obs)
        else:
            self.state.rollouts.insert(
                observations=obs,
                recurrent_hidden_states=self.state.recurrent_hidden_states,
                actions=self.state.action,
                action_log_probs=self.state.actor_critic_output.distributions.log_probs(
                    self.state.action
                ),
                value_preds=self.state.actor_critic_output.values,
                rewards=torch.tensor([[0.0]], dtype=torch.float32, device=self.device),
                masks=torch.tensor([[1.0]], dtype=torch.float32, device=self.device),
            )
            self.state.rollouts.after_update()

        with torch.no_grad():
            step_observation = {
                k: v[self.state.rollouts.step]
                for k, v in self.state.rollouts.observations.items()
            }

            (
                self.state.actor_critic_output,
                self.state.recurrent_hidden_states,
            ) = self.actor_critic(
                step_observation,
                self.state.rollouts.recurrent_hidden_states[self.state.rollouts.step],
                self.state.rollouts.prev_actions[self.state.rollouts.step],
                self.state.rollouts.masks[self.state.rollouts.step],
            )

        self.state.action = self.state.actor_critic_output.distributions.mode()

    @property
    def action_name(self):
        return self._actions[self.state.action[0].item()]

    def mapped_action(self):
        action = self.action_name
        return action if action != "End" else "Stop"

    def step(self, event):
        self.update_state(event)
        return self.mapped_action()
