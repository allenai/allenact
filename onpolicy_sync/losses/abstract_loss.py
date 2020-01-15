import abc

from onpolicy_sync.storage import RolloutStorage
from rl_base.common import Loss, ActorCriticOutput
from rl_base.distributions import CategoricalDistr


class AbstractActorCriticLoss(Loss):
    @abc.abstractmethod
    def loss(
        self,
        rollouts: RolloutStorage,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        raise NotImplementedError()
