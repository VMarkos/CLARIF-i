"""Curriculum sorting environment aligned with SortingEnv lessons.

Uses factored MultiDiscrete (i, j) swaps, discrete permutation observations,
Kendall-tau potential shaping with a mild living cost, and horizons that scale
with the active list size whenever the curriculum stage changes.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import kendalltau


def infer_active_n(state: np.ndarray, max_n: int) -> int:
    """Infer active prefix length from identity-padded discrete state."""
    for n in range(max_n, 1, -1):
        if not np.array_equal(state[n:], np.arange(n, max_n)):
            continue
        if set(state[:n].tolist()) == set(range(n)):
            return n
    return 2


class CurriculumSortEnv(gym.Env):
    """Mixed-stage curriculum env with SortingEnv-style spaces and rewards."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_n: int = 10, start_stage: int = 2):
        super().__init__()
        self.max_n = max_n
        self.max_stage = start_stage
        self.current_n = start_stage
        self.steps = 0
        self.max_steps = self._horizon_for(start_stage)
        self.state = np.arange(self.max_n, dtype=np.int64)
        self.goal = np.arange(self.max_n, dtype=np.int64)
        self._prev_tau = -1.0

        self.observation_space = spaces.MultiDiscrete([max_n] * max_n)
        self.action_space = spaces.MultiDiscrete([max_n, max_n])

        # Retained for older call sites that mapped unordered pairs -> flat ids.
        self.swap_pairs = [
            (i, j) for i in range(max_n) for j in range(i + 1, max_n)
        ]
        self.pair_to_action = {pair: idx for idx, pair in enumerate(self.swap_pairs)}

    @staticmethod
    def _horizon_for(n: int) -> int:
        return max(100, int(n) ** 2)

    def set_max_stage(self, stage: int) -> None:
        """Advance curriculum ceiling and keep the step budget in sync."""
        self.max_stage = min(int(stage), self.max_n)
        # Ensure the horizon scales with the new stage (fixes frozen-horizon pitfall).
        self.max_steps = self._horizon_for(self.max_stage)

    def action_masks(self) -> np.ndarray:
        """Mask inactive indices in both swap slots (MaskablePPO / SortingEnv style)."""
        masks_i = np.zeros(self.max_n, dtype=bool)
        masks_i[: self.current_n] = True
        return np.concatenate([masks_i, masks_i])

    def _get_tau(self) -> float:
        tau, _ = kendalltau(self.state[: self.current_n], self.goal[: self.current_n])
        return -1.0 if np.isnan(tau) else float(tau)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Mix sampling: 70% current max_stage, 30% a strictly smaller stage.
        if self.max_stage > 2 and self.np_random.random() < 0.3:
            self.current_n = int(self.np_random.integers(2, self.max_stage))
        else:
            self.current_n = self.max_stage

        self.max_steps = self._horizon_for(self.current_n)

        self.state = np.arange(self.max_n, dtype=np.int64)
        active = np.arange(self.current_n, dtype=np.int64)
        self.np_random.shuffle(active)
        self.state[: self.current_n] = active

        self._prev_tau = self._get_tau()
        return self.state.copy(), {}

    def step(self, action):
        self.steps += 1
        i = int(action[0])
        j = int(action[1])

        if i >= self.current_n or j >= self.current_n:
            obs = self.state.copy()
            return obs, -2.0, True, False, {"is_success": False}

        if i == j:
            reward = -0.25
            terminated = self._is_sorted_active()
            truncated = self.steps >= self.max_steps
            is_success = terminated
            if terminated:
                reward += 10.0
            elif truncated:
                reward -= 1.0
            return self.state.copy(), float(reward), terminated, truncated, {
                "is_success": is_success
            }

        self.state[i], self.state[j] = self.state[j], self.state[i]
        tau = self._get_tau()
        reward = (tau - self._prev_tau) - 0.005
        self._prev_tau = tau

        terminated = self._is_sorted_active()
        truncated = self.steps >= self.max_steps
        is_success = terminated

        if terminated:
            reward += 10.0
        elif truncated:
            reward -= 1.0

        return self.state.copy(), float(reward), terminated, truncated, {
            "is_success": is_success
        }

    def _is_sorted_active(self) -> bool:
        return bool(np.array_equal(self.state[: self.current_n], self.goal[: self.current_n]))

    def action_to_swap(self, action) -> tuple[int, int]:
        return int(action[0]), int(action[1])

    def get_current_array(self) -> np.ndarray:
        return self.state[: self.current_n].copy()

    @staticmethod
    def is_sorted(arr: np.ndarray) -> bool:
        return bool(np.all(arr[:-1] <= arr[1:]))

    @staticmethod
    def sorted_percentage(arr: np.ndarray) -> float:
        sorted_arr = np.sort(arr.copy())
        correct = len(arr) - np.count_nonzero(arr - sorted_arr)
        return correct / len(arr)
