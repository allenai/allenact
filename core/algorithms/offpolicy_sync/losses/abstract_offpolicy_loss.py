"""Defining abstract loss classes for actor critic models."""

import abc
import typing
from typing import Dict, Tuple

import torch

from core.algorithms.onpolicy_sync.policy import ObservationType
from core.base_abstractions.misc import Loss, Memory

ModelType = typing.TypeVar("ModelType")


class AbstractOffPolicyLoss(typing.Generic[ModelType], Loss):
    """Abstract class representing an off-policy loss function used to train a
    model."""

    @abc.abstractmethod
    def loss(  # type: ignore
        self, model: ModelType, batch: ObservationType, memory: Memory, *args, **kwargs,
    ) -> Tuple[torch.FloatTensor, Dict[str, float], Memory, int]:
        """Computes the loss.

        Loss after processing a batch of data with (part of) a model (possibly with memory).

        # Parameters

        model: model to run on data batch (both assumed to be on the same device)
        batch: data to use as input for model (already on the same device as model)
        memory: model memory before processing current data batch

        # Returns

        A tuple with:

        current_loss: total loss
        current_info: additional information about the current loss
        memory: model memory after processing current data batch
        bsize: batch size
        """
        raise NotImplementedError()
