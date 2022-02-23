"""Defining abstract loss classes for actor critic models."""

import abc
from typing import Dict, Tuple, Union

import torch

from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import Loss, ActorCriticOutput


class AbstractActorCriticLoss(Loss):
    """Abstract class representing a loss function used to train an
    ActorCriticModel."""

    # noinspection PyMethodOverriding
    @abc.abstractmethod
    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ) -> Union[
        Tuple[torch.FloatTensor, Dict[str, float]],
        Tuple[torch.FloatTensor, Dict[str, float], Dict[str, float]],
    ]:
        """Computes the loss.

        # Parameters

        batch : A batch of data corresponding to the information collected when rolling out (possibly many) agents
            over a fixed number of steps. In particular this batch should have the same format as that returned by
            `RolloutStorage.batched_experience_generator`.
        actor_critic_output : The output of calling an ActorCriticModel on the observations in `batch`.
        args : Extra args.
        kwargs : Extra kwargs.

        # Returns

        A (0-dimensional) torch.FloatTensor corresponding to the computed loss. `.backward()` will be called on this
        tensor in order to compute a gradient update to the ActorCriticModel's parameters.
        A Dict[str, float] with scalar values corresponding to sub-losses.
        An optional Dict[str, float] with scalar values corresponding to extra info to be processed per epoch and
        combined across epochs by the engine.
        """
        # TODO: The above documentation is missing what the batch dimensions are.

        raise NotImplementedError()
