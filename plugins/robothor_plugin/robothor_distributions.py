from typing import Tuple, cast

import torch
from gym import spaces

from core.base_abstractions.distributions import CategoricalDistr, Distr
from utils import spaces_utils as su


class TupleCategoricalDistr(Distr):
    def __init__(self, probs=None, logits=None, validate_args=None):
        if probs is not None:
            cats = [
                CategoricalDistr(probs=probs[..., it, :], validate_args=validate_args)
                for it in range(probs.shape[-2])
            ]
        else:
            cats = [
                CategoricalDistr(logits=logits[..., it, :], validate_args=validate_args)
                for it in range(logits.shape[-2])
            ]

        self.dists = tuple(cats)

        self.num_agents = len(self.dists)

        nactions = self.dists[0].param_shape[-1]
        self.action_space = spaces.Tuple((spaces.Discrete(nactions),) * self.num_agents)

        self.log_prob_space = su.log_prob_space(self.action_space)

    def log_probs(self, actions: Tuple[torch.LongTensor, ...]) -> torch.FloatTensor:
        return su.flatten(
            self.log_prob_space,
            tuple(
                [
                    dist.log_probs(act).squeeze(-1)  # added action dim
                    for dist, act in zip(self.dists, actions)
                ]
            ),
        )

    def entropy(self) -> torch.FloatTensor:
        return cast(
            torch.FloatTensor,
            torch.stack([dist.entropy() for dist in self.dists], dim=-1),
        )

    def sample(self, sample_shape=torch.Size()) -> Tuple[torch.LongTensor, ...]:
        return tuple([dist.sample(sample_shape) for dist in self.dists])

    def mode(self) -> Tuple[torch.LongTensor, ...]:
        return tuple([dist.mode() for dist in self.dists])
