# rl/AlgorithmicSortingEnv.py
#
# Environment that defines rewards based on specific sorting algorithms

from numpy import ndarray, int32, full, random, concatenate
from typing import Callable
from gymnasium import spaces
from lnai.rl.SortingEnv import SortingEnv
from lnai.rl.SortingCoach import SortingCoach

class AlgorithmicSortingEnv(SortingEnv):
    def __init__(self, algorithm: Callable[[ndarray], ndarray], p_advice: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.coach = SortingCoach(algorithm)
        self.p_advice = p_advice

        self.observation_space = spaces.Box(
            low=-1,
            high=self.max_n,
            shape=(2 * self.max_n, ),
            dtype=int32
        )


    def _get_advice_vector(self) -> ndarray:
        '''Returns advice vector as an ndarray instead of a 2-tuple'''
        advice_vec = full(self.max_n, -1, dtype=int32)
        if random.rand() < self.p_advice:
            advice = self.coach.get_advice(self.state[:self.n])
            if advice is not None:
                advice_vec[advice[0]] = 1
                advice_vec[advice[1]] = 1
        return advice_vec
        

    def reset(self, seed=None, options=None) -> tuple[ndarray, dict]:
        _, info = super().reset(seed, options)
        self.coach.update_state(self.state[:self.n])
        return self._get_obs(), info
        

    def step(self, action) -> tuple:
        _, reward, terminated, truncated, info = super().step(action)
        return self._get_obs(), reward, terminated, truncated, info


    def _get_obs(self) -> ndarray:
        return concatenate([self.state, self._get_advice_vector()])


    '''
    def _get_reward(self, i: int, j: int) -> float:
        reward = super()._get_reward(i, j)
        advice = self.coach.get_advice(self.state)
        if advice is None:
            return reward + self.out_of_algorithm_penalty
        dist = abs(min(advice) - min(i, j)) + abs(max(advice) - max(i, j))
        reward += 1.0 / (1 + dist)
        return reward
    '''

