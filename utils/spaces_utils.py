# Original work Copyright (c) 2016 OpenAI (https://openai.com).
# Modified work Copyright (c) Allen Institute for AI
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Tuple, List
from collections import OrderedDict

import numpy as np
import torch
from gym import spaces as gym

from core.algorithms.onpolicy_sync.policy import ActionType


def flatdim(space):
    """Return the number of dimensions a flattened equivalent of this space
    would have.

    Accepts a space and returns an integer. Raises ``NotImplementedError`` if
    the space is not defined in ``gym.spaces``.
    """
    if isinstance(space, gym.Box):
        return int(np.prod(space.shape))
    elif isinstance(space, gym.Discrete):
        return 1  # we do not expand to one-hot
    elif isinstance(space, gym.Tuple):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, gym.Dict):
        return int(sum([flatdim(s) for s in space.spaces.values()]))
    elif isinstance(space, gym.MultiBinary):
        return int(space.n)
    elif isinstance(space, gym.MultiDiscrete):
        return int(np.prod(space.shape))
    else:
        raise NotImplementedError


def flatten(space, torch_x):
    """Flatten data points from a space.
    """
    if isinstance(space, gym.Box):
        if len(space.shape) > 0:
            return torch_x.view(torch_x.shape[: -len(space.shape)] + (-1,))
        else:
            return torch_x.view(torch_x.shape + (-1,))
    elif isinstance(space, gym.Discrete):
        # Assume tensor input does NOT contain a dimension for action
        if isinstance(torch_x, torch.Tensor):
            return torch_x.unsqueeze(-1)
        else:
            return torch.tensor(torch_x).view(1)
    elif isinstance(space, gym.Tuple):
        return torch.cat(
            [flatten(s, x_part) for x_part, s in zip(torch_x, space.spaces)], dim=-1
        )
    elif isinstance(space, gym.Dict):
        return torch.cat(
            [flatten(s, torch_x[key]) for key, s in space.spaces.items()], dim=-1
        )
    elif isinstance(space, gym.MultiBinary):
        return torch_x.view(torch_x.shape[: -len(space.shape)] + (-1,))
    elif isinstance(space, gym.MultiDiscrete):
        return torch_x.view(torch_x.shape[: -len(space.shape)] + (-1,))
    else:
        raise NotImplementedError


def unflatten(space, torch_x):
    """Unflatten a concatenated data points tensor from a space.
    """
    if isinstance(space, gym.Box):
        return torch_x.view(torch_x.shape[:-1] + space.shape).float()
    elif isinstance(space, gym.Discrete):
        res = torch_x.view(torch_x.shape[:-1] + space.shape).long()
        return res if len(res.shape) > 0 else res.item()
    elif isinstance(space, gym.Tuple):
        dims = [flatdim(s) for s in space.spaces]
        list_flattened = torch.split(torch_x, dims, dim=-1)
        list_unflattened = [
            unflatten(s, flattened)
            for flattened, s in zip(list_flattened, space.spaces)
        ]
        return tuple(list_unflattened)
    elif isinstance(space, gym.Dict):
        dims = [flatdim(s) for s in space.spaces.values()]
        list_flattened = torch.split(torch_x, dims, dim=-1)
        list_unflattened = [
            (key, unflatten(s, flattened))
            for flattened, (key, s) in zip(list_flattened, space.spaces.items())
        ]
        return OrderedDict(list_unflattened)
    elif isinstance(space, gym.MultiBinary):
        return torch_x.view(torch_x.shape[:-1] + space.shape).byte()
    elif isinstance(space, gym.MultiDiscrete):
        return torch_x.view(torch_x.shape[:-1] + space.shape).long()
    else:
        raise NotImplementedError


def torch_point(space, np_x):
    """Convert numpy space point into torch.
    """
    if isinstance(space, gym.Box):
        return torch.from_numpy(np_x)
    elif isinstance(space, gym.Discrete):
        return np_x
    elif isinstance(space, gym.Tuple):
        return tuple([torch_point(s, x_part) for x_part, s in zip(np_x, space.spaces)])
    elif isinstance(space, gym.Dict):
        return OrderedDict(
            [(key, torch_point(s, np_x[key])) for key, s in space.spaces.items()]
        )
    elif isinstance(space, gym.MultiBinary):
        return torch.from_numpy(np_x)
    elif isinstance(space, gym.MultiDiscrete):
        return torch.from_numpy(np_x)
    else:
        raise NotImplementedError


def numpy_point(
    space: gym.Space, torch_x: Union[int, torch.Tensor, OrderedDict, Tuple]
):
    """Convert torch space point into numpy.
    """
    if isinstance(space, gym.Box):
        return torch_x.cpu().numpy()
    elif isinstance(space, gym.Discrete):
        return torch_x
    elif isinstance(space, gym.Tuple):
        return tuple(
            [numpy_point(s, x_part) for x_part, s in zip(torch_x, space.spaces)]
        )
    elif isinstance(space, gym.Dict):
        return OrderedDict(
            [(key, numpy_point(s, torch_x[key])) for key, s in space.spaces.items()]
        )
    elif isinstance(space, gym.MultiBinary):
        return torch_x.cpu().numpy()
    elif isinstance(space, gym.MultiDiscrete):
        return torch_x.cpu().numpy()
    else:
        raise NotImplementedError


def flatten_space(space: gym.Space):
    if isinstance(space, gym.Box):
        return gym.Box(space.low.flatten(), space.high.flatten())
    if isinstance(space, gym.Discrete):
        return gym.Box(low=0, high=space.n, shape=(1,))
    if isinstance(space, gym.Tuple):
        space = [flatten_space(s) for s in space.spaces]
        return gym.Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, gym.Dict):
        space = [flatten_space(s) for s in space.spaces.values()]
        return gym.Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, gym.MultiBinary):
        return gym.Box(low=0, high=1, shape=(space.n,))
    if isinstance(space, gym.MultiDiscrete):
        return gym.Box(low=np.zeros_like(space.nvec), high=space.nvec,)
    raise NotImplementedError


def log_prob_space(space: gym.Space):
    if isinstance(space, gym.Box):
        return gym.Box(low=-np.inf, high=np.inf, shape=space.shape)
    if isinstance(space, gym.Discrete):
        return gym.Box(low=-np.inf, high=0.0, shape=())
    if isinstance(space, gym.Tuple):
        return gym.Tuple(list(log_prob_space(s) for s in space.spaces))
    if isinstance(space, gym.Dict):
        return gym.Dict({key: log_prob_space(space[key]) for key in space.spaces})
    if isinstance(space, gym.MultiBinary):
        return gym.Box(low=-np.inf, high=0.0, shape=(space.n,))
    if isinstance(space, gym.MultiDiscrete):
        return gym.Box(low=-np.inf, high=0.0, shape=(len(space.nvec),))
    raise NotImplementedError


def value_space(space: gym.Space):
    if isinstance(space, gym.Box):
        return gym.Box(low=-np.inf, high=np.inf, shape=())
    if isinstance(space, gym.Discrete):
        return gym.Box(low=-np.inf, high=np.inf, shape=())
    if isinstance(space, gym.Tuple):
        return gym.Tuple(list(value_space(s) for s in space.spaces))
    if isinstance(space, gym.Dict):
        return gym.Dict({key: value_space(space[key]) for key in space.spaces})
    if isinstance(space, gym.MultiBinary):
        return gym.Box(low=-np.inf, high=np.inf, shape=())
    if isinstance(space, gym.MultiDiscrete):
        return gym.Box(low=-np.inf, high=np.inf, shape=())
    raise NotImplementedError


def action_list(action_space: gym.Space, actions: ActionType) -> List[ActionType]:
    # no step dimension in actions
    return [unflatten(action_space, ac) for ac in flatten(action_space, actions)]
