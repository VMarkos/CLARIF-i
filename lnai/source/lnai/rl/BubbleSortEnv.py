# BubbleSortEnv.py
#
# Environment that fosters bubble sort learning

from numpy import array, ndarray, zeros
from gymnasium import spaces
from utils import bubble_sort
from lnai.rl.AlgorithmicSortingEnv import AlgorithmicSortingEnv

class BubbleSortEnv(AlgorithmicSortingEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(bubble_sort, *args, **kwargs)
        # Each action `i` corresponds to the swap ``i`` <-> `i + 1`
        self.action_space = spaces.Discrete(self.max_n - 1)
        self._accurate_steps = 0


    def step(self, action: int) -> tuple:
        swap_action = array([action, action + 1])
        return super().step(swap_action)


    def action_masks(self) -> ndarray:
        mask = zeros(self.max_n - 1, dtype=bool)
        mask[:self.n - 1] = True
        return mask


    def _get_reward(self, i: int, j: int) -> float:
        tau = self._get_tau()
        delta = tau - self._prev_tau
        self._prev_tau = tau
        reward = delta
        if self._is_terminated():
            reward += (12.0 - (self.n - self._start_size) / (self.max_n - self._start_size))# * self._accurate_steps / self._ticks
            return reward
        reward -= 0.05
        prev_state = self.state.copy()
        prev_state[i], prev_state[j] = prev_state[i], prev_state[j]
        advice = self.coach.get_advice(prev_state)
        if advice is None:
            return reward - 0.5
        if delta > 0 and i in advice and j in advice:
            reward += 1.0
            self._accurate_steps += 1
        return reward


    def reset(self, seed=None, options=None) -> tuple:
        obs, info = super().reset(seed, options)
        self._max_steps = self.n * (self.n - 1) // 2 + 5
        self._accurate_steps = 0
        return obs, info
