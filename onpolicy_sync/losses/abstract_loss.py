"""Defining abstract loss classes for actor critic models."""

import abc
from typing import Dict, Union, Tuple

import torch

from rl_base.common import Loss, ActorCriticOutput
from rl_base.distributions import CategoricalDistr


class AbstractActorCriticLoss(Loss):
    """Abstract class representing a loss function used to train an
    ActorCriticModel."""

    @abc.abstractmethod
    def loss(  # type: ignore
        self,
        step_count: int,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        """Computes the loss.

        # Parameters

        batch : A batch of data corresponding to the information collected when rolling out (possibly many) agents
            over a fixed number of steps. In particular this batch should have the same format as that returned by
            `RolloutStorage.recurrent_generator`.
        actor_critic_output : The output of calling an ActorCriticModel on the observations in `batch`.
        args : Extra args.
        kwargs : Extra kwargs.

        # Returns

        A (0-dimensional) torch.FloatTensor corresponding to the computed loss. `.backward()` will be called on this
        tensor in order to compute a gradient update to the ActorCriticModel's parameters.
        """
        # TODO: The above documentation is missing what the batch dimensions are.

        raise NotImplementedError()
