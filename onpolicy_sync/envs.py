import abc
import os

import gym
from gym.spaces.box import Box

try:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    from baselines import bench

    # noinspection PyPackageRequirements,PyUnresolvedReferences
    from baselines.common.atari_wrappers import make_atari, wrap_deepmind

    BASELINES_IMPORTED = True
except ImportError:
    BASELINES_IMPORTED = False

try:
    import dm_control2gym
except ImportError:
    pass

try:
    # noinspection PyPackageRequirements
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    assert (
        BASELINES_IMPORTED
    ), "Cannot run `make_env` as the baseline library could not be imported."

    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split(".")
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        # noinspection PyUnresolvedReferences
        is_atari = hasattr(gym.envs, "atari") and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv
        )
        if is_atari:
            env = make_atari(env_id)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets,
            )

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example."
            )

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        # noinspection PyProtectedMember
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info["bad_transition"] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.next_task(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        # noinspection PyProtectedMember
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper, abc.ABC):
    def __init__(self, env=None):
        """Transpose observation space (base class)"""
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=(2, 0, 1)):
        """Transpose observation space for images."""
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])
