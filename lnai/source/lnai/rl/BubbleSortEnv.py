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
            reward += 10.0
        reward -= 0.05
        advice = self.coach.get_advice(self.state)
        if advice is None:
            return reward - 0.5
        if delta > 0 and i in advice and j in advice:
            reward += 1.0
        return reward


    def reset(self, seed=None, options=None) -> tuple:
        obs, info = super().reset(seed, options)
        self._max_steps = self.n * (self.n - 1) // 2 + 5
        return obs, info
