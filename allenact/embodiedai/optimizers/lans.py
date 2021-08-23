# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
from torch.optim import Optimizer
import numpy as np


@functools.wraps(print)
def print_r0(*args, **kwargs):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


class Lans(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=1e-4,
        min_trust=0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.min_trust = min_trust
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Lamb does not support sparse gradients, consider SparseAdam instad."
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                grad = grad / (torch.norm(grad) + group["eps"])

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = 1.0 / (exp_avg_sq.sqrt() + group["eps"])

                adam_step = exp_avg * denom
                lans_step = grad * denom

                if group["weight_decay"] != 0:
                    adam_step.add_(p.data, alpha=group["weight_decay"])
                    lans_step.add_(p.data, alpha=group["weight_decay"])

                weight_norm = torch.norm(p.data).item()
                adam_step_norm = torch.norm(adam_step).item()
                lans_step_norm = torch.norm(lans_step).item()

                if weight_norm == 0 or adam_step_norm == 0:
                    adam_trust_ratio = 1
                else:
                    adam_trust_ratio = np.clip(weight_norm, 0, 10) / adam_step_norm

                if weight_norm == 0 or lans_step_norm == 0:
                    lans_trust_ratio = 1
                else:
                    lans_trust_ratio = np.clip(weight_norm, 0, 10) / lans_step_norm

                state["weight_norm"] = weight_norm
                state["lans_step_norm"] = lans_step_norm
                state["lans_trust_ratio"] = lans_trust_ratio
                state["adam_step_norm"] = adam_step_norm
                state["adam_trust_ratio"] = adam_trust_ratio
                state["second_moment_norm"] = torch.norm(exp_avg_sq.sqrt()).item()
                state["first_moment_norm"] = torch.norm(exp_avg).item()

                if self.min_trust > 0:
                    adam_trust_ratio = np.clip(
                        adam_trust_ratio, self.min_trust, 1.0 / self.min_trust
                    )
                    lans_trust_ratio = np.clip(
                        lans_trust_ratio, self.min_trust, 1.0 / self.min_trust
                    )

                step_size = group["lr"]

                p.data.add_(adam_step, alpha=-step_size * beta1 * adam_trust_ratio)
                p.data.add_(
                    lans_step, alpha=-step_size * (1 - beta1) * lans_trust_ratio
                )

        return loss
