# CurriculumPPO_AnnealedBC

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from lnai.rl.CurriculumSortEnv import infer_active_n


class CurriculumPPO_AnnealedBC(MaskablePPO):
    """MaskablePPO with annealed behavioural cloning toward a swap expert."""

    def __init__(self, *args, expert=None, initial_bc_coef: float = 0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert = expert
        self.initial_bc_coef = initial_bc_coef
        self.max_n = self.env.get_attr("max_n")[0]

    def _prepare_actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.long()
        if isinstance(self.action_space, spaces.MultiDiscrete):
            if actions.ndim == 1:
                actions = actions.view(-1, int(self.action_space.nvec.shape[0]))
            return actions
        return actions.flatten()

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        current_bc_coef = self.initial_bc_coef * self._current_progress_remaining

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = self._prepare_actions(rollout_data.actions)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                value_loss = F.mse_loss(rollout_data.returns, values)
                entropy_loss = -torch.mean(entropy)

                obs_cpu = rollout_data.observations.cpu().numpy()
                teacher_actions = []
                for obs in obs_cpu:
                    state = np.asarray(obs).reshape(-1)
                    # Obs is the discrete permutation (no trailing action channel).
                    current_n = infer_active_n(state, self.max_n)
                    action = self.expert.get_action(state, current_n)
                    teacher_actions.append(np.asarray(action, dtype=np.int64))

                teacher_actions_tensor = torch.as_tensor(
                    np.stack(teacher_actions),
                    device=self.device,
                    dtype=torch.long,
                )
                teacher_actions_tensor = self._prepare_actions(teacher_actions_tensor)

                _, target_log_prob, _ = self.policy.evaluate_actions(
                    rollout_data.observations,
                    teacher_actions_tensor,
                    action_masks=rollout_data.action_masks,
                )
                bc_loss = -torch.mean(target_log_prob)

                total_loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                    + current_bc_coef * bc_loss
                )

                self.policy.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

        self._n_updates += self.n_epochs
