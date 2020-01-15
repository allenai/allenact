from .a2c_acktr import A2C_ACKTR
from .ppo import PPO


class A2C(A2C_ACKTR):
    def __init__(
        self,
        actor_critic,
        value_loss_coef,
        entropy_coef,
        optimizer,
        max_grad_norm=None,
    ):
        super().__init__(
            actor_critic,
            value_loss_coef,
            entropy_coef,
            optimizer,
            max_grad_norm,
            acktr=False,
        )


class ACKTR(A2C_ACKTR):
    def __init__(
        self,
        actor_critic,
        value_loss_coef,
        entropy_coef,
        optimizer,
        max_grad_norm=None,
    ):
        super().__init__(
            actor_critic,
            value_loss_coef,
            entropy_coef,
            optimizer,
            max_grad_norm,
            acktr=True,
        )
