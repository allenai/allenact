import math
import random
from collections import defaultdict
from typing import Union, Tuple, Optional, Dict, Callable

import torch
import torch.nn.functional as F

from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.algorithms.onpolicy_sync.storage import MiniBatchExperienceStorage
from allenact.base_abstractions.misc import GenericAbstractLoss, ModelType, Memory
from allenact.utils.misc_utils import unzip, partition_sequence


def _index_recursive(d: Dict, key: Union[str, Tuple[str, ...]]):
    if isinstance(key, str):
        return d[key]
    for k in key:
        d = d[k]
    return d


class InverseDynamicsVDRLoss(GenericAbstractLoss):
    def __init__(
        self,
        compute_action_logits_fn: Callable,
        img0_key: str,
        img1_key: str,
        action_key: str,
    ):
        self.compute_action_logits_fn = compute_action_logits_fn
        self.img0_key = img0_key
        self.img1_key = img1_key
        self.action_key = action_key

    def loss(
        self,
        *,
        model: ModelType,
        batch: ObservationType,
        batch_memory: Memory,
        stream_memory: Memory,
    ) -> Tuple[torch.Tensor, Dict[str, float], Memory, Memory, int]:
        action_logits = self.compute_action_logits_fn(
            model=model, img0=batch[self.img0_key], img1=batch[self.img1_key],
        )
        loss = F.cross_entropy(action_logits, target=batch[self.action_key])
        return (
            loss,
            {"cross_entropy": loss.item()},
            batch_memory,
            stream_memory,
            int(batch[self.img0_key].shape[0]),
        )


class DiscreteVisualDynamicsReplayStorage(MiniBatchExperienceStorage):
    def __init__(
        self,
        image_uuid: Union[str, Tuple[str, ...]],
        action_success_uuid: Optional[Union[str, Tuple[str, ...]]],
        nactions: int,
        num_to_store_per_action: int,
        max_to_save_per_episode: int,
        target_batch_size: int,
    ):
        self.image_uuid = image_uuid
        self.action_success_uuid = action_success_uuid
        self.nactions = nactions
        self.num_to_store_per_action = num_to_store_per_action
        self.max_to_save_per_episode = max_to_save_per_episode
        self.target_batch_size = target_batch_size

        self._prev_imgs: Optional[
            Dict[Union[str, Tuple[str, ...]], torch.Tensor]
        ] = None

        self.action_to_saved_transitions = {i: [] for i in range(nactions)}
        self.action_to_num_seen = {i: 0 for i in range(nactions)}
        self.task_sampler_to_actions_already_sampled = defaultdict(lambda: set())

        self.device = torch.device("cpu")

    def set_partition(self, index: int, num_parts: int):
        self.num_to_store_per_action = math.ceil(
            self.num_to_store_per_action / num_parts
        )
        self.target_batch_size = math.ceil(self.target_batch_size / num_parts)

    def initialize(self, observations: ObservationType):
        self._prev_imgs = None
        self.add(observations=observations, actions=None, masks=None)

    def batched_experience_generator(self, num_mini_batch: int):
        triples = [
            (i0, a, i1)
            for a, v in self.action_to_saved_transitions.items()
            for (i0, i1) in v
        ]
        random.shuffle(triples)

        if len(triples) == 0:
            return

        parts = partition_sequence(
            triples, math.ceil(len(triples) / self.target_batch_size)
        )
        for part in parts:
            img0s, actions, img1s = unzip(part, n=3)

            img0 = torch.stack([i0.to(self.device) for i0 in img0s], 0)
            action = torch.tensor(actions, device=self.device)
            img1 = torch.stack([i1.to(self.device) for i1 in img1s], 0)

            yield {"img0": img0, "action": action, "img1": img1}

    def add(
        self,
        *,
        observations: ObservationType,
        actions: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        **kwargs,
    ):
        cur_imgs = _index_recursive(d=observations, key=self.image_uuid).cpu()

        if self._prev_imgs is not None:
            actions = actions.view(-1).cpu().numpy()
            masks = masks.view(-1).cpu().numpy()

            if self.action_success_uuid is not None:
                action_successes = (
                    observations[self.action_success_uuid].cpu().view(-1).numpy()
                )
            else:
                action_successes = [True] * actions.shape[0]

            nsamplers = actions.shape[0]
            assert nsamplers == masks.shape[0]

            for i, (a, m, action_success) in enumerate(
                zip(actions, masks, action_successes)
            ):
                actions_already_sampled_in_ep = self.task_sampler_to_actions_already_sampled[
                    i
                ]

                if (
                    m != 0
                    and action_success
                    and (
                        len(actions_already_sampled_in_ep)
                        <= self.max_to_save_per_episode
                    )
                    and a not in actions_already_sampled_in_ep
                ):  # Not the start of a new episode/task -> self._prev_imgs[i] corresponds to cur_imgs[i]
                    saved_transitions = self.action_to_saved_transitions[a]

                    if len(saved_transitions) < self.num_to_store_per_action:
                        saved_transitions.append((self._prev_imgs[i], cur_imgs[i]))
                    else:
                        saved_transitions[
                            random.randint(0, len(saved_transitions) - 1)
                        ] = (
                            self._prev_imgs[i],
                            cur_imgs[i],
                        )

                    # Reservoir sampling transitions
                    # a = int(a)
                    # saved_transitions = self.action_to_saved_transitions[a]
                    # num_seen = self.action_to_num_seen[a]
                    # if num_seen < self.triples_to_save_per_action:
                    #     saved_transitions.append((self._prev_imgs[i], cur_imgs[i]))
                    # else:
                    #     index = random.randint(0, num_seen)
                    #     if index < self.triples_to_save_per_action:
                    #         saved_transitions[index] = (self._prev_imgs[i], cur_imgs[i])

                    actions_already_sampled_in_ep.add(a)
                    self.action_to_num_seen[a] += 1
                else:
                    actions_already_sampled_in_ep.clear()

        self._prev_imgs = cur_imgs

    def before_update(self, **kwargs):
        pass

    def after_update(self, **kwargs):
        pass

    def to(self, device: torch.device):
        self.device = device
