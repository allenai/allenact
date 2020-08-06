"""Defining abstract loss classes for actor critic models."""

import abc
import typing
from typing import Dict, Union, Tuple

import torch

from core.base_abstractions.misc import Loss

ModelType = typing.TypeVar("ModelType")


class AbstractOffPolicyLoss(typing.Generic[ModelType], Loss):
    """Abstract class representing a loss function used to train an
    ActorCriticModel."""

    @abc.abstractmethod
    def loss(  # type: ignore
        self,
        model: ModelType,
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
        memory: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Dict[str, float], Dict[str, torch.Tensor]]:
        """Computes the loss.

        # TODO: Description of how this works
        """
        raise NotImplementedError()
