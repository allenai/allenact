import abc
from typing import Dict, Union, Tuple

import torch

from rl_base.common import Loss, ActorCriticOutput
from rl_base.distributions import CategoricalDistr


class AbstractActorCriticLoss(Loss):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def loss(  # type: ignore
        self,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        raise NotImplementedError()
