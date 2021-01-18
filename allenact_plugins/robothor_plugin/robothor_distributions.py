from typing import Tuple

import torch

from allenact.base_abstractions.distributions import CategoricalDistr, Distr


class TupleCategoricalDistr(Distr):
    def __init__(self, probs=None, logits=None, validate_args=None):
        self.dists = CategoricalDistr(
            probs=probs, logits=logits, validate_args=validate_args
        )

    def log_prob(self, actions: Tuple[torch.LongTensor, ...]) -> torch.FloatTensor:
        # flattened output [steps, samplers, num_agents]
        return self.dists.log_prob(torch.stack(actions, dim=-1))

    def entropy(self) -> torch.FloatTensor:
        # flattened output [steps, samplers, num_agents]
        return self.dists.entropy()

    def sample(self, sample_shape=torch.Size()) -> Tuple[torch.LongTensor, ...]:
        # split and remove trailing singleton dim
        res = self.dists.sample(sample_shape).split(1, dim=-1)
        return tuple([r.view(r.shape[:2]) for r in res])

    def mode(self) -> Tuple[torch.LongTensor, ...]:
        # split and remove trailing singleton dim
        res = self.dists.mode().split(1, dim=-1)
        return tuple([r.view(r.shape[:2]) for r in res])
