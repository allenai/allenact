import abc
from typing import Dict, Union

import torch

from rl_base.common import Loss, ActorCriticOutput
from rl_base.distributions import CategoricalDistr


class AbstractActorCriticLoss(Loss):
    @abc.abstractmethod
    def loss(
        self,
        batch: Dict[str, Union[Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        raise NotImplementedError()
