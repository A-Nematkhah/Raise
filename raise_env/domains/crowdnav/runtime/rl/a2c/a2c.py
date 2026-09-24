"""Minimal A2C optimizer matching the PPO agent interface used by train.py."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


class A2C:
    """
    Classic n-step A2C (no PPO clipping). Same ``update(rollouts)`` surface
    as ``rl.ppo.PPO`` so Stage II can swap agents without changing the loop.
    """

    def __init__(
        self,
        actor_critic,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        alpha=0.99,
        max_grad_norm=None,
    ):
        self.actor_critic = actor_critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), lr=lr, eps=eps, alpha=alpha
        )

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]

        # Must match Policy/SRNN ``nminibatch`` (nenv // nminibatch at train time).
        base = getattr(self.actor_critic, "base", None)
        num_mini_batch = int(getattr(base, "nminibatch", 1) or 1)
        num_mini_batch = max(1, num_mini_batch)

        if self.actor_critic.is_recurrent:
            data_generator = rollouts.recurrent_generator(
                advantages, num_mini_batch=num_mini_batch
            )
        else:
            data_generator = rollouts.feed_forward_generator(
                advantages, num_mini_batch=num_mini_batch
            )

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        n = 0

        for sample in data_generator:
            (
                obs_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                _value_preds_batch,
                return_batch,
                masks_batch,
                _old_action_log_probs_batch,
                adv_targ,
            ) = sample

            values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                obs_batch,
                recurrent_hidden_states_batch,
                masks_batch,
                actions_batch,
            )

            action_loss = -(adv_targ.detach() * action_log_probs).mean()
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

            self.optimizer.zero_grad()
            (
                value_loss * self.value_loss_coef
                + action_loss
                - dist_entropy * self.entropy_coef
            ).backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()
            n += 1

        return (
            value_loss_epoch / max(n, 1),
            action_loss_epoch / max(n, 1),
            dist_entropy_epoch / max(n, 1),
        )
