# Adapted from OpenAI's gym.spaces.utils

from collections import OrderedDict

import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary, Tuple, Dict


def flatdim(space):
    """Return the number of dimensions a flattened equivalent of this space
    would have.

    Accepts a space and returns an integer. Raises ``NotImplementedError`` if
    the space is not defined in ``gym.spaces``.
    """
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return 1  # does not expand to one-hot
    elif isinstance(space, Tuple):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([flatdim(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return int(np.prod(space.shape))
    else:
        raise NotImplementedError


def flatten(space, x):
    """Flatten data points from a space.
    """
    if isinstance(space, Box):
        return x.view(x.shape[: -len(space.shape)] + (-1,))
    elif isinstance(space, Discrete):
        # assume tensor input already contains a dimension for action
        return x if isinstance(x, torch.Tensor) else torch.tensor(x).view(-1)
    elif isinstance(space, Tuple):
        return torch.cat(
            [flatten(s, x_part) for x_part, s in zip(x, space.spaces)], dim=-1
        )
    elif isinstance(space, Dict):
        return torch.cat(
            [flatten(s, x[key]) for key, s in space.spaces.items()], dim=-1
        )
    elif isinstance(space, MultiBinary):
        return x.view(x.shape[: -len(space.shape)] + (-1,))
    elif isinstance(space, MultiDiscrete):
        return x.view(x.shape[: -len(space.shape)] + (-1,))
    else:
        raise NotImplementedError


def unflatten(space, x):
    """Unflatten a concatenated data points tensor from a space.
    """
    if isinstance(space, Box):
        return x.view(x.shape[:-1] + space.shape).float()
    elif isinstance(space, Discrete):
        res = x.view(x.shape[:-1] + space.shape).long()
        return res if len(res.shape) > 0 else res.item()
    elif isinstance(space, Tuple):
        dims = [flatdim(s) for s in space.spaces]
        list_flattened = torch.split(x, dims, dim=-1)
        list_unflattened = [
            unflatten(s, flattened)
            for flattened, s in zip(list_flattened, space.spaces)
        ]
        return tuple(list_unflattened)
    elif isinstance(space, Dict):
        dims = [flatdim(s) for s in space.spaces.values()]
        list_flattened = torch.split(x, dims, dim=-1)
        list_unflattened = [
            (key, unflatten(s, flattened))
            for flattened, (key, s) in zip(list_flattened, space.spaces.items())
        ]
        return OrderedDict(list_unflattened)
    elif isinstance(space, MultiBinary):
        return x.view(x.shape[:-1] + space.shape).byte()
    elif isinstance(space, MultiDiscrete):
        return x.view(x.shape[:-1] + space.shape).long()
    else:
        raise NotImplementedError


def torch_point(space, x):
    """Convert numpy space point into torch.
    """
    if isinstance(space, Box):
        return torch.from_numpy(x)
    elif isinstance(space, Discrete):
        return x
    elif isinstance(space, Tuple):
        return tuple([torch_point(s, x_part) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, Dict):
        return OrderedDict(
            [(key, torch_point(s, x[key])) for key, s in space.spaces.items()]
        )
    elif isinstance(space, MultiBinary):
        return torch.from_numpy(x)
    elif isinstance(space, MultiDiscrete):
        return torch.from_numpy(x)
    else:
        raise NotImplementedError
