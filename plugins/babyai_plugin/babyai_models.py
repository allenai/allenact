import typing
from typing import Dict, Optional, List

import babyai.model
import babyai.rl
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces.dict import Dict as SpaceDict
from torch import Tensor

from common.algorithms.onpolicy_sync.policy import ActorCriticModel
from common.base_abstractions.misc import ActorCriticOutput
from common.base_abstractions.distributions import CategoricalDistr


class BabyAIACModelWrapped(babyai.model.ACModel):
    def __init__(
        self,
        obs_space: Dict[str, int],
        action_space: gym.spaces.Discrete,
        image_dim=128,
        memory_dim=128,
        instr_dim=128,
        use_instr=False,
        lang_model="gru",
        use_memory=False,
        arch="cnn1",
        aux_info=None,
        include_auxiliary_head: bool = False,
    ):
        self.use_cnn2 = arch == "cnn2"
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            image_dim=image_dim,
            memory_dim=memory_dim,
            instr_dim=instr_dim,
            use_instr=use_instr,
            lang_model=lang_model,
            use_memory=use_memory,
            arch="cnn1" if self.use_cnn2 else arch,
            aux_info=aux_info,
        )

        self.semantic_embedding = None
        if self.use_cnn2:
            self.semantic_embedding = nn.Embedding(33, embedding_dim=8)
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=24, out_channels=16, kernel_size=(2, 2)),
                *self.image_conv[1:]
            )
            self.image_conv[0].apply(babyai.model.initialize_parameters)

        self.include_auxiliary_head = include_auxiliary_head
        if self.use_memory and self.lang_model == "gru":
            self.memory_rnn = nn.LSTM(self.image_dim, self.memory_dim)

        if self.include_auxiliary_head:
            self.aux = nn.Sequential(
                nn.Linear(self.memory_dim, 64), nn.Tanh(), nn.Linear(64, action_space.n)
            )
            self.aux.apply(babyai.model.initialize_parameters)

        self.train()

    def forward_once(self, obs, memory, instr_embedding=None):
        """Copied (with minor modifications) from
        `babyai.model.ACModel.forward(...)`."""
        if self.use_instr and instr_embedding is None:
            instr_embedding = self._get_instr_embedding(obs.instr)
        if self.use_instr and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            mask = (obs.instr != 0).float()
            # The mask tensor has the same length as obs.instr, and
            # thus can be both shorter and longer than instr_embedding.
            # It can be longer if instr_embedding is computed
            # for a subbatch of obs.instr.
            # It can be shorter if obs.instr is a subbatch of
            # the batch that instr_embeddings was computed for.
            # Here, we make sure that mask and instr_embeddings
            # have equal length along dimension 1.
            mask = mask[:, : instr_embedding.shape[1]]
            instr_embedding = instr_embedding[:, : mask.shape[1]]

            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        if self.arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            for controler in self.controllers:
                x = controler(x, instr_embedding)
            x = F.relu(self.film_pool(x))
        else:
            x = self.image_conv(x.contiguous())

        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (
                memory[:, : self.semi_memory_size],
                memory[:, self.semi_memory_size :],
            )
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_instr and not "filmcnn" in self.arch:
            embedding = torch.cat((embedding, instr_embedding), dim=1)

        if hasattr(self, "aux_info") and self.aux_info:
            extra_predictions = {
                info: self.extra_heads[info](embedding) for info in self.extra_heads
            }
        else:
            extra_predictions = dict()

        return {
            "embedding": embedding,
            "memory": memory,
            "extra_predictions": extra_predictions,
        }

    def forward_loop(
        self,
        observations: Dict[str, torch.Tensor],
        recurrent_hidden_states: torch.Tensor,
        prev_actions: torch.LongTensor,
        masks: torch.FloatTensor,
        **kwargs
    ):
        results = []
        images = observations["minigrid_ego_image"].float()
        instrs: Optional[torch.Tensor] = None
        if "minigrid_mission" in observations:
            instrs = observations["minigrid_mission"]

        _, nsamplers, _ = recurrent_hidden_states.shape
        rollouts_len = images.shape[0] // nsamplers
        obs = babyai.rl.DictList()

        images = images.view(rollouts_len, nsamplers, *images.shape[1:])
        masks = masks.view(rollouts_len, nsamplers, *masks.shape[1:])

        # needs_reset = (masks != 1.0).view(nrollouts, -1).any(-1)
        if instrs is not None:
            instrs = instrs.view(rollouts_len, nsamplers, instrs.shape[-1])

        needs_instr_reset_mask = masks != 1.0
        needs_instr_reset_mask[0] = 1
        needs_instr_reset_mask: Tensor = needs_instr_reset_mask.squeeze(-1)
        instr_embeddings: Optional[torch.Tensor] = None
        if self.use_instr:
            instr_reset_multi_inds = list(
                (int(a), int(b))
                for a, b in zip(*np.where(needs_instr_reset_mask.cpu().numpy()))
            )
            time_ind_to_which_need_instr_reset = [[] for _ in range(rollouts_len)]
            reset_multi_ind_to_index = {
                mi: i for i, mi in enumerate(instr_reset_multi_inds)
            }
            for a, b in instr_reset_multi_inds:
                time_ind_to_which_need_instr_reset[a].append(b)

            unique_instr_embeddings = self._get_instr_embedding(
                instrs[needs_instr_reset_mask]
            )

            instr_embeddings_list = []
            instr_embeddings_list.append(unique_instr_embeddings[:nsamplers])
            current_instr_embeddings_list = list(instr_embeddings_list[-1])

            for time_ind in range(1, rollouts_len):
                if len(time_ind_to_which_need_instr_reset[time_ind]) == 0:
                    instr_embeddings_list.append(instr_embeddings_list[-1])
                else:
                    for sampler_needing_reset_ind in time_ind_to_which_need_instr_reset[
                        time_ind
                    ]:
                        current_instr_embeddings_list[
                            sampler_needing_reset_ind
                        ] = unique_instr_embeddings[
                            reset_multi_ind_to_index[
                                (time_ind, sampler_needing_reset_ind)
                            ]
                        ]

                    instr_embeddings_list.append(
                        torch.stack(current_instr_embeddings_list, dim=0)
                    )

            instr_embeddings = torch.stack(instr_embeddings_list, dim=0)

        assert recurrent_hidden_states.shape[0] == 1
        memory = recurrent_hidden_states[0]
        # instr_embedding: Optional[torch.Tensor] = None
        for i in range(rollouts_len):
            obs.image = images[i]
            if "minigrid_mission" in observations:
                obs.instr = instrs[i]

            # reset = needs_reset[i].item()
            # if self.baby_ai_model.use_instr and (reset or i == 0):
            #     instr_embedding = self.baby_ai_model._get_instr_embedding(obs.instr)

            results.append(
                self.forward_once(
                    obs, memory=memory * masks[i], instr_embedding=instr_embeddings[i]
                )
            )
            memory = results[-1]["memory"]

        embedding = torch.cat([r["embedding"] for r in results], dim=0)

        extra_predictions_list = [r["extra_predictions"] for r in results]
        extra_predictions = {
            key: torch.cat([ep[key] for ep in extra_predictions_list], dim=0)
            for key in extra_predictions_list[0]
        }
        return (
            ActorCriticOutput(
                distributions=CategoricalDistr(logits=self.actor(embedding),),
                values=self.critic(embedding),
                extras=extra_predictions
                if not self.include_auxiliary_head
                else {
                    **extra_predictions,
                    "auxiliary_distributions": CategoricalDistr(
                        logits=self.aux(embedding)
                    ),
                },
            ),
            torch.stack([r["memory"] for r in results], dim=0),
        )

    # noinspection PyMethodOverriding
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        recurrent_hidden_states: torch.Tensor,
        prev_actions: torch.LongTensor,
        masks: torch.FloatTensor,
        **kwargs
    ):
        if self.lang_model != "gru":
            return self.forward_loop(
                observations=observations,
                recurrent_hidden_states=recurrent_hidden_states,
                prev_actions=prev_actions,
                masks=masks,
                **kwargs
            )

        assert recurrent_hidden_states.shape[0] == 1

        images = observations["minigrid_ego_image"]
        if self.use_cnn2:
            images_shape = images.shape
            images = images + torch.LongTensor([0, 11, 22]).view(1, 1, 1, 3).to(
                images.device
            )
            images = self.semantic_embedding(images).view(*images_shape[:3], 24)
        images = images.permute(0, 3, 1, 2).float()

        _, nsamplers, _ = recurrent_hidden_states.shape
        rollouts_len = images.shape[0] // nsamplers

        masks = typing.cast(
            torch.FloatTensor, masks.view(rollouts_len, nsamplers, *masks.shape[1:])
        )
        instrs: Optional[torch.Tensor] = None
        if "minigrid_mission" in observations and self.use_instr:
            instrs = observations["minigrid_mission"]
            instrs = instrs.view(rollouts_len, nsamplers, instrs.shape[-1])

        needs_instr_reset_mask = masks != 1.0
        needs_instr_reset_mask[0] = 1
        needs_instr_reset_mask: Tensor = needs_instr_reset_mask.squeeze(-1)
        blocking_inds: List[int] = np.where(
            needs_instr_reset_mask.view(rollouts_len, -1).any(-1).cpu().numpy()
        )[0].tolist()
        blocking_inds.append(rollouts_len)

        instr_embeddings: Optional[torch.Tensor] = None
        if self.use_instr:
            instr_reset_multi_inds = list(
                (int(a), int(b))
                for a, b in zip(*np.where(needs_instr_reset_mask.cpu().numpy()))
            )
            time_ind_to_which_need_instr_reset = [[] for _ in range(rollouts_len)]
            reset_multi_ind_to_index = {
                mi: i for i, mi in enumerate(instr_reset_multi_inds)
            }
            for a, b in instr_reset_multi_inds:
                time_ind_to_which_need_instr_reset[a].append(b)

            unique_instr_embeddings = self._get_instr_embedding(
                instrs[needs_instr_reset_mask]
            )

            instr_embeddings_list = []
            instr_embeddings_list.append(unique_instr_embeddings[:nsamplers])
            current_instr_embeddings_list = list(instr_embeddings_list[-1])

            for time_ind in range(1, rollouts_len):
                if len(time_ind_to_which_need_instr_reset[time_ind]) == 0:
                    instr_embeddings_list.append(instr_embeddings_list[-1])
                else:
                    for sampler_needing_reset_ind in time_ind_to_which_need_instr_reset[
                        time_ind
                    ]:
                        current_instr_embeddings_list[
                            sampler_needing_reset_ind
                        ] = unique_instr_embeddings[
                            reset_multi_ind_to_index[
                                (time_ind, sampler_needing_reset_ind)
                            ]
                        ]

                    instr_embeddings_list.append(
                        torch.stack(current_instr_embeddings_list, dim=0)
                    )

            instr_embeddings = torch.stack(instr_embeddings_list, dim=0)

        # The following code can be used to compute the instr_embeddings in another way
        # and thus verify that the above logic is (more likely to be) correct
        # needs_instr_reset_mask = (masks != 1.0)
        # needs_instr_reset_mask[0] *= 0
        # needs_instr_reset_inds = needs_instr_reset_mask.view(nrollouts, -1).any(-1).cpu().numpy()
        #
        # # Get inds where a new task has started
        # blocking_inds: List[int] = np.where(needs_instr_reset_inds)[0].tolist()
        # blocking_inds.append(needs_instr_reset_inds.shape[0])
        # if nrollouts != 1:
        #     pdb.set_trace()
        # if blocking_inds[0] != 0:
        #     blocking_inds.insert(0, 0)
        # if self.use_instr:
        #     instr_embeddings_list = []
        #     for ind0, ind1 in zip(blocking_inds[:-1], blocking_inds[1:]):
        #         instr_embeddings_list.append(
        #             self._get_instr_embedding(instrs[ind0])
        #             .unsqueeze(0)
        #             .repeat(ind1 - ind0, 1, 1)
        #         )
        #     tmp_instr_embeddings = torch.cat(instr_embeddings_list, dim=0)
        # assert (instr_embeddings - tmp_instr_embeddings).abs().max().item() < 1e-6

        # Embed images
        # images = images.view(nrollouts, nsamplers, *images.shape[1:])
        image_embeddings = self.image_conv(images)
        if self.arch.startswith("expert_filmcnn"):
            instr_embeddings_flatter = instr_embeddings.view(
                -1, *instr_embeddings.shape[2:]
            )
            for controller in self.controllers:
                image_embeddings = controller(
                    image_embeddings, instr_embeddings_flatter
                )
            image_embeddings = F.relu(self.film_pool(image_embeddings))

        image_embeddings = image_embeddings.view(rollouts_len, nsamplers, -1)

        if self.use_instr and self.lang_model == "attgru":
            raise NotImplementedError("Currently attgru is not implemented.")

        memory = None
        if self.use_memory:
            assert recurrent_hidden_states.shape[0] == 1
            hidden = (
                recurrent_hidden_states[:, :, : self.semi_memory_size],
                recurrent_hidden_states[:, :, self.semi_memory_size :],
            )
            embeddings_list = []
            for ind0, ind1 in zip(blocking_inds[:-1], blocking_inds[1:]):
                hidden = (hidden[0] * masks[ind0], hidden[1] * masks[ind0])
                rnn_out, hidden = self.memory_rnn(image_embeddings[ind0:ind1], hidden)
                embeddings_list.append(rnn_out)

            # embedding = hidden[0]
            embedding = torch.cat(embeddings_list, dim=0)
            memory = torch.cat(hidden, dim=-1)
        else:
            embedding = image_embeddings

        if self.use_instr and not "filmcnn" in self.arch:
            embedding = torch.cat((embedding, instr_embeddings), dim=-1)

        if hasattr(self, "aux_info") and self.aux_info:
            extra_predictions = {
                info: self.extra_heads[info](embedding) for info in self.extra_heads
            }
        else:
            extra_predictions = dict()

        embedding = embedding.view(rollouts_len * nsamplers, -1)

        return (
            ActorCriticOutput(
                distributions=CategoricalDistr(logits=self.actor(embedding),),
                values=self.critic(embedding),
                extras=extra_predictions
                if not self.include_auxiliary_head
                else {
                    **extra_predictions,
                    "auxiliary_distributions": CategoricalDistr(
                        logits=self.aux(embedding)
                    ),
                },
            ),
            memory,
        )


class BabyAIRecurrentACModel(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        image_dim=128,
        memory_dim=128,
        instr_dim=128,
        use_instr=False,
        lang_model="gru",
        use_memory=False,
        arch="cnn1",
        aux_info=None,
        include_auxiliary_head: bool = False,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        assert "minigrid_ego_image" in observation_space.spaces
        assert not use_instr or "minigrid_mission" in observation_space.spaces

        self.memory_dim = memory_dim
        self.include_auxiliary_head = include_auxiliary_head

        self.baby_ai_model = BabyAIACModelWrapped(
            obs_space={"image": 7 * 7 * 3, "instr": 100,},
            action_space=action_space,
            image_dim=image_dim,
            memory_dim=memory_dim,
            instr_dim=instr_dim,
            use_instr=use_instr,
            lang_model=lang_model,
            use_memory=use_memory,
            arch=arch,
            aux_info=aux_info,
            include_auxiliary_head=self.include_auxiliary_head,
        )

    @property
    def recurrent_hidden_state_size(self) -> int:
        return 2 * self.memory_dim

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(  # type: ignore
        self,
        observations: Dict[str, torch.Tensor],
        recurrent_hidden_states: torch.Tensor,
        prev_actions: torch.LongTensor,
        masks: torch.FloatTensor,
        **kwargs
    ):
        return self.baby_ai_model.forward(
            observations=observations,
            recurrent_hidden_states=recurrent_hidden_states,
            prev_actions=prev_actions,
            masks=masks,
            **kwargs
        )
