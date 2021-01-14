from typing import Tuple, Union, cast

import torch

from core.base_abstractions.distributions import CategoricalDistr, Distr


class TupleCategoricalDistr(Distr):
    def __init__(self, probs=None, logits=None, validate_args=None):
        if probs is not None:
            params = [
                CategoricalDistr(probs=probs[..., it, :], validate_args=validate_args)
                for it in range(probs.shape[-2])
            ]
        else:
            params = [
                CategoricalDistr(logits=logits[..., it, :], validate_args=validate_args)
                for it in range(logits.shape[-2])
            ]

        self.dists = tuple(params)

    def log_prob(self, actions: Tuple[torch.LongTensor, ...]) -> torch.FloatTensor:
        return cast(
            torch.FloatTensor,
            torch.stack(
                [dist.log_prob(act) for dist, act in zip(self.dists, actions)], dim=-1
            ),
        )

    def entropy(self) -> torch.FloatTensor:
        return cast(
            torch.FloatTensor,
            torch.stack([dist.entropy() for dist in self.dists], dim=-1),
        )
        # # Independent sources => joint entropy == sum of entropies
        # res: Union[int, torch.FloatTensor] = 0
        # for dist in self.dists:
        #     res = res + dist.entropy()
        # return res

    def sample(self, sample_shape=torch.Size()) -> Tuple[torch.LongTensor, ...]:
        return tuple([dist.sample(sample_shape) for dist in self.dists])

    def mode(self) -> Tuple[torch.LongTensor, ...]:
        return tuple([dist.mode() for dist in self.dists])
