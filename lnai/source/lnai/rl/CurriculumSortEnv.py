import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces



# -------------------------------------------------------------------
# 1. Environment with Mixed-Stage Resets & Action Masking
# -------------------------------------------------------------------
class CurriculumSortEnv(gym.Env):
    def __init__(self, max_n: int = 10):
        super().__init__()
        self.max_n = max_n
        self.max_stage = 2  # Upper bound for curriculum sampling

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.max_n + 1,), dtype=np.float32
        )

        self.swap_pairs = list(itertools.combinations(range(self.max_n), 2))
        self.pair_to_action = {pair: idx for idx, pair in enumerate(self.swap_pairs)}
        self.action_space = spaces.Discrete(len(self.swap_pairs))

        self.state = np.zeros(self.max_n, dtype=np.float32)
        self.last_action = -1.0
        self.steps = 0
        self.current_n = 2

    def set_max_stage(self, stage: int):
        self.max_stage = min(stage, self.max_n)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_action = -1.0
        self.state = np.zeros(self.max_n, dtype=np.float32)

        # Mix sampling: 70% chance to sample current max_stage, 30% lower stages
        if self.max_stage > 2 and np.random.rand() < 0.3:
            self.current_n = int(np.random.randint(2, self.max_stage))
        else:
            self.current_n = self.max_stage

        self.max_steps = 2 * self.current_n ** 2

        # Sample and normalize state values into [-1.0, 1.0] range
        raw_vals = np.random.uniform(-10.0, 10.0, size=(self.current_n,))
        normalized_vals = raw_vals / 10.0

        self.state[:self.current_n] = normalized_vals
        self.state[self.current_n:] = 0.0  # Use neutral 0.0 padding instead of large 999.0

        obs = np.append(self.state, self.last_action / len(self.swap_pairs))
        return obs.astype(np.float32), {}

    def step(self, action: int):
        self.steps += 1
        i, j = self.swap_pairs[action]

        # Invalid action check for elements outside current_n
        if i >= self.current_n or j >= self.current_n:
            reward = -2.0
            terminated = True
            obs = np.append(self.state, self.last_action / len(self.swap_pairs))
            return obs.astype(np.float32), reward, terminated, False, {"is_success": False}

        repeat_penalty = -0.2 if action == self.last_action else 0.0

        # Execute Swap
        self.state[i], self.state[j] = self.state[j], self.state[i]
        self.last_action = float(action)

        active_slice = self.state[:self.current_n]
        is_sorted = bool(np.all(active_slice[:-1] <= active_slice[1:]))

        reward = repeat_penalty

        if is_sorted:
            reward += max(5.0, 2.0 * self.current_n)
            terminated = True
            is_success = True
        elif self.steps >= self.max_steps:
            reward += -1.0
            terminated = True
            is_success = False
        else:
            # Reward shaping: give small potential reward if Kendall-Tau distance decreased
            reward += max(-0.05, -1.0 / self.current_n)
            terminated = False
            is_success = False

        obs = np.append(self.state, self.last_action / len(self.swap_pairs))
        return obs.astype(np.float32), reward, terminated, False, {"is_success": is_success}


    def action_to_swap(self, action: int) -> tuple[int, int]:
        """Maps discrete action index to (i, j) array indices to swap."""
        return self.swap_pairs[action]

    def get_current_array(self) -> np.ndarray:
        """Returns the current raw array state for coach evaluation."""
        return self.state.copy()

    @staticmethod
    def is_sorted(arr: np.ndarray) -> bool:
        """Checks if the array is sorted in ascending order."""
        return np.all(arr[:-1] <= arr[1:])


    @staticmethod
    def sorted_percentage(arr: np.ndarray) -> float:
        '''Returns the percentage of array elements that are sorted'''
        sorted_arr = np.sort(arr.copy())
        correct = len(arr) - np.count_nonzero(arr - sorted_arr)
        return correct / len(arr)
