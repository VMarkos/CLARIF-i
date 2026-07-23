# rl/CoachedSortingEnv.py
#
# Coaching based sorting environment, where a coach helps by masking invalid cases.

from numpy import ndarray, zeros, concatenate
from typing import Callable
from lnai.rl.SortingEnv import SortingEnv
from lnai.rl.SortingCoach import SortingCoach

class CoachedSortingEnv(SortingEnv):
    def __init__(self, algorithm, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.coach = SortingCoach(algorithm)
        self._advice = None


    def reset(self, seed=None, options=None) -> tuple[ndarray, dict]:
        obs, info = super().reset(seed, options)
        self.coach.update_state(self.state[:self.n])
        return obs, info


    def _update_advice(self) -> None:
        self._advice = self.coach.get_advice(self.state)


    def step(self, action) -> tuple:
        self._update_advice()
        return super().step(action)


    def _get_reward(self, i: int, j: int) -> float:
        reward = super()._get_reward(i, j)
        delta = self._get_tau() - self._prev_tau
        if delta > 0 and self._advice is not None and i in self._advice and j in self._advice:
            reward += 1.0
        return reward

    
    def action_masks(self) -> ndarray:
        if self._advice is None:
            return super().action_masks()
        mask_min = zeros(self.max_n, dtype=bool)
        mask_min[min(self._advice)] = True
        mask_max = zeros(self.max_n, dtype=bool)
        mask_max[max(self._advice)] = True
        return concatenate([mask_min, mask_max])
