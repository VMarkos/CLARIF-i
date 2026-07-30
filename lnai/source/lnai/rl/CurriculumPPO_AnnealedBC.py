# CurriculumPPO_AnnealedBC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO

class CurriculumPPO_AnnealedBC(PPO):
    def __init__(self, *args, expert=None, initial_bc_coef: float = 0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert = expert
        self.initial_bc_coef = initial_bc_coef
        self.max_n = self.env.get_attr('max_n')[0]

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)

        # Anneal BC coefficient down as training progresses
        current_bc_coef = self.initial_bc_coef * self._current_progress_remaining

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                value_loss = F.mse_loss(rollout_data.returns, values)
                entropy_loss = -torch.mean(entropy)

                # BC Coaching Loss computation
                obs_cpu = rollout_data.observations.cpu().numpy()
                teacher_actions = []
                for obs in obs_cpu:
                    raw_state = obs[:-1]
                    # Estimate active n from non-zero values or bounds
                    active_elements = np.where(raw_state != 0.0)[0]
                    nonzero_count = np.count_nonzero(raw_state)
                    current_n = max(2, active_elements[-1] + 1) if len(active_elements) > 0 else 2
                    current_n = min(self.max_n, current_n)
                    action = self.expert.get_action(obs, current_n)
                    teacher_actions.append(action)

                teacher_actions_tensor = torch.tensor(
                    teacher_actions, device=self.device, dtype=torch.long
                )

                _, target_log_prob, _ = self.policy.evaluate_actions(
                    rollout_data.observations, teacher_actions_tensor
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
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        self._n_updates += self.n_epochs
