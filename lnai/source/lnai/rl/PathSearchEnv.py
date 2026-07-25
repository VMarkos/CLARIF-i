# rl/PathSearchEnv.py
#
# Path searching formulation of sorting

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CurriculumPathSearchEnv(gym.Env):
    """
    PathSearchEnv supporting Curriculum Learning and Action Masking.
    Fixed max capacity N_max, but dynamically adjusts current level n <= N_max.
    """
    def __init__(self, max_n=8, current_n=3, max_path_len=30):
        super().__init__()
        self.max_n = max_n
        self.max_path_len = max_path_len
        self.current_n = min(current_n, max_n)

        # Global Action Space fixed to MAX capacity: [Swap_0, ..., Swap_{max_n-2}, BACKTRACK]
        self.max_swaps = max_n - 1
        self.action_space = spaces.Discrete(self.max_swaps + 1)
        self.BACKTRACK_ACTION = self.max_swaps

        # Swaps for MAX capacity
        self.all_swaps = [(i, i + 1) for i in range(max_n - 1)]

        # Fixed Observation Space padded to max_n
        self.observation_space = spaces.Dict({
            "current_array": spaces.Box(low=-1, high=max_n-1, shape=(max_n,), dtype=np.int32),
            "active_mask": spaces.Box(low=0, high=1, shape=(max_n,), dtype=np.int32), # 1 for real elements, 0 for padding
            "action_mask": spaces.Box(low=0, high=1, shape=(self.max_swaps + 1,), dtype=np.int8)
        })

        self.reset()

    def set_curriculum_level(self, new_n: int):
        """Allows external trainer to ramp up difficulty (e.g., from n=3 to n=8)."""
        assert 2 <= new_n <= self.max_n, f"Level must be between 2 and {self.max_n}"
        self.current_n = new_n

    def action_masks(self) -> np.ndarray:
        """
        Returns a boolean mask of shape (max_swaps + 1,).
        Used directly by MaskablePPO / sb3-contrib wrappers.
        """
        mask = np.zeros(self.max_swaps + 1, dtype=bool)

        # 1. Enable swaps valid for CURRENT n
        active_swaps_count = self.current_n - 1
        mask[:active_swaps_count] = True

        # 2. Backtrack Action Validity
        if len(self.path_stack) > 1:
            mask[self.BACKTRACK_ACTION] = True
        else:
            mask[self.BACKTRACK_ACTION] = False

        return mask

    def _get_obs(self):
        curr_arr = self.path_stack[-1]
        
        # Pad current state up to max_n with -1
        padded_array = np.full(self.max_n, -1, dtype=np.int32)
        padded_array[:self.current_n] = curr_arr

        # Active mask indicates which positions matter
        active_mask = np.zeros(self.max_n, dtype=np.int32)
        active_mask[:self.current_n] = 1

        return {
            "current_array": padded_array,
            "active_mask": active_mask,
            "action_mask": self.action_masks().astype(np.int8)
        }

    def _get_inversion_count(self, arr):
        inv = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    inv += 1
        return inv

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Scramble array of current size n
        initial_arr = np.random.permutation(self.current_n)
        self.target_state = np.arange(self.current_n)
        
        self.path_stack = [initial_arr]
        self.visited_states = {tuple(initial_arr)}
        self.current_inversions = self._get_inversion_count(initial_arr)

        return self._get_obs(), {}

    def step(self, action):
        # Enforce Mask Safety
        mask = self.action_masks()
        if not mask[action]:
            # Invalid action executed -> Heavy penalty
            return self._get_obs(), -1.0, False, False, {"invalid_action": True}

        terminated = False
        truncated = False
        reward = 0.0

        if action == self.BACKTRACK_ACTION:
            self.path_stack.pop()
            new_state = self.path_stack[-1]
            new_inv = self._get_inversion_count(new_state)
            reward = -0.05 + 0.1 * (self.current_inversions - new_inv)
            self.current_inversions = new_inv
        else:
            i, j = self.all_swaps[action]
            curr = self.path_stack[-1].copy()
            curr[i], curr[j] = curr[j], curr[i]
            
            state_tuple = tuple(curr)
            if state_tuple in self.visited_states:
                reward = -0.2
            else:
                self.path_stack.append(curr)
                self.visited_states.add(state_tuple)
                new_inv = self._get_inversion_count(curr)
                reward = 0.5 * (self.current_inversions - new_inv) - 0.01
                self.current_inversions = new_inv

                if np.array_equal(curr, self.target_state):
                    reward += 10.0
                    terminated = True

        if len(self.path_stack) >= self.max_path_len:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}
