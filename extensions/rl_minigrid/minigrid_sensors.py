import typing
from typing import Optional, Any, Dict

import gym
import gym_minigrid.minigrid
import numpy as np
import torch
from babyai.utils.format import InstructionsPreprocessor
from gym_minigrid.minigrid import MiniGridEnv

from rl_base.sensor import Sensor
from rl_base.task import Task, SubTaskType

# fmt: off
ALL_VOCAB_TOKENS = [
    "a", "after", "and", "ball", "behind", "blue", "box",
    "door", "front", "go", "green", "grey", "in", "key",
    "left", "next", "of", "on", "open", "pick", "purple",
    "put", "red", "right", "the", "then", "to", "up", "yellow",
    "you", "your",
]
# fmt: on


class EgocentricMiniGridSensor(Sensor[MiniGridEnv, Task[MiniGridEnv]]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)
        self.agent_view_size = config["agent_view_size"]
        self.view_channels = config.get("view_channels", 1)
        self.num_objects = (
            typing.cast(
                int, max(map(abs, gym_minigrid.minigrid.OBJECT_TO_IDX.values()))
            )
            + 1
        )
        self.num_colors = (
            typing.cast(int, max(map(abs, gym_minigrid.minigrid.COLOR_TO_IDX.values())))
            + 1
        )
        self.num_states = (
            typing.cast(int, max(map(abs, gym_minigrid.minigrid.STATE_TO_IDX.values())))
            + 1
        )
        self.observation_space = self._get_observation_space()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "minigrid_ego_image"

    def _get_observation_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=0,
            high=max(self.num_objects, self.num_colors, self.num_states) - 1,
            shape=(self.agent_view_size, self.agent_view_size, self.view_channels),
            dtype=int,
        )

    def get_observation(
        self,
        env: MiniGridEnv,
        task: Optional[SubTaskType],
        *args,
        minigrid_output_obs: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Any:
        if minigrid_output_obs is not None and minigrid_output_obs["image"].shape == (
            self.agent_view_size,
            self.agent_view_size,
        ):
            img = minigrid_output_obs["image"][:, :, : self.view_channels]
        else:
            env.agent_view_size = self.agent_view_size
            img = env.gen_obs()["image"][:, :, : self.view_channels]

        assert img.dtype == np.uint8
        return img


class MiniGridMissionSensor(Sensor[MiniGridEnv, Task[MiniGridEnv]]):
    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(config, *args, **kwargs)

        self.instr_preprocessor = InstructionsPreprocessor(
            model_name="TMP_SENSOR", load_vocab_from=None
        )

        # We initialize the vocabulary with a fixed collection of tokens
        # and then ensure that the size cannot exceed this number. This
        # guarantees that sensors on all processes will produce the same
        # values.
        for token in ALL_VOCAB_TOKENS:
            self.instr_preprocessor.vocab[token]
        self.instr_preprocessor.vocab.max_size = len(ALL_VOCAB_TOKENS)

        self.instr_len: int = self.config["instr_len"]
        self.observation_space = self._get_observation_space()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "minigrid_mission"

    def _get_observation_space(self) -> gym.Space:
        return gym.spaces.Box(
            low=0,
            high=self.instr_preprocessor.vocab.max_size,
            shape=(self.instr_len,),
            dtype=int,
        )

    def get_observation(
        self,
        env: MiniGridEnv,
        task: Optional[SubTaskType],
        *args,
        minigrid_output_obs: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> Any:
        if minigrid_output_obs is None:
            minigrid_output_obs = env.gen_obs()

        out = self.instr_preprocessor([minigrid_output_obs]).view(-1)

        n: int = out.shape[0]
        if n > self.instr_len:
            out = out[: self.instr_len]
        elif n < self.instr_len:
            out = torch.nn.functional.pad(
                input=out, pad=[0, self.instr_len - n], value=0,
            )

        return out.long().numpy()
