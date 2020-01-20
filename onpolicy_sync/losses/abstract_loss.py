import abc
from typing import Dict, Union

import torch

from rl_base.common import Loss, ActorCriticOutput
from rl_base.distributions import CategoricalDistr


class AbstractActorCriticLoss(Loss):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def loss(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        raise NotImplementedError()
