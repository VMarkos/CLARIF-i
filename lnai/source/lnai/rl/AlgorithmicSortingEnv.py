# rl/AlgorithmicSortingEnv.py
#
# Environment that defines rewards based on specific sorting algorithms

from numpy import ndarray
from typing import Callable
from lnai.rl.SortingEnv import SortingEnv
from lnai.rl.SortingCoach import SortingCoach

class AlgorithmicSortingEnv(SortingEnv):
    def __init__(self, algorithm: Callable[[ndarray], ndarray], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.coach = SortingCoach(algorithm)
        self.out_of_algorithm_penalty = -0.1
        

    def reset(self, seed=None, options=None) -> tuple[ndarray, dict]:
        obs, info = super().reset(seed, options)
        self.coach.update_state(self.state[:self.n])
        return obs, info
        

    '''
    def step(self, action) -> tuple:
        advice = self.coach.get_advice(self.state)
        i, j = action[0], action[1]
        penalty = 0.0
        if advice is None:
            penalty = -0.5
        obs, _, terminated, truncated, info = super().step(action)
        reward = self._get_reward(i, j, action) + penalty
        return obs, reward, terminated, truncated, info
    '''

    def _get_reward(self, i: int, j: int) -> float:
        reward = super()._get_reward(i, j)
        advice = self.coach.get_advice(self.state)
        if advice is None:
            return reward + self.out_of_algorithm_penalty
        dist = abs(min(advice) - min(i, j)) + abs(max(advice) - max(i, j))
        reward += 1.0 / (1 + dist)
        return reward


    def set_n(self, new_n: int) -> None:
        super().set_n(new_n)
        self.out_of_algorithm_penalty = -1.0 + 1 / self.n
