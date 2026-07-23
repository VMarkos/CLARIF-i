# rl/Environment.py

import numpy as np
from gymnasium import Env, spaces
from matplotlib import pyplot as plt
from scipy.stats import kendalltau

import random

# Should this be elsewhere?
# random.seed(5679813004)


class SortingEnv(Env):
    def __init__(self, max_n: int, start_size: int=4, target_steps: int | None=None) -> None:
        super(SortingEnv, self).__init__()
        
        # Instance "globals"
        self.render_mode = 'ansi'

        # Initialize object fields
        self.n = start_size
        self.max_n = max_n
        self._ticks = 0
        self._max_steps = max(self.n ** 2, 100)
        #if target_steps is None:
        #    self._target_steps = self.n ** 2 // 2 # self.n * np.log(self.n)
        #else:
        #    self._target_steps = target_steps

        # Initialize state
        #self.__init_state()

        # Discrete obesrvation space containing n^n (not all valid) different objects
        self.observation_space = spaces.MultiDiscrete([max_n] * max_n)

        # Action space where each action is a swap
        self.action_space = spaces.MultiDiscrete([max_n, max_n])



    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Resets internal state"""
        super().reset(seed=seed, options=options)

        # State is an ndarray
        self.state = np.arange(self.max_n)
        active = np.random.permutation(self.n)
        self.state[:self.n] = active

        # GOAL is a sorted ndarray
        self.GOAL = np.arange(self.max_n)

        # Previous tau value
        self._prev_tau = self._get_tau()

        self._ticks = 0
        # self._max_steps = self.n * (self.n - 1) // 2 + 5#max(self.n ** 2, 100)

        return self.state.copy(), dict()



    def _get_tau(self) -> float:
        '''Failsafe kendall's tau computation'''
        tau, _ = kendalltau(self.state[:self.n], self.GOAL[:self.n])
        return -1.0 if np.isnan(tau) else tau


    def render(self) -> str | None:
        """Renders the environment as ANSI string."""
        print(self.state)
        return str(self.state)


    def step(self, action) -> tuple:
        """Makes a step forward by applying an action to the current setting"""
        i, j = action[0], action[1]
        self.state[i], self.state[j] = self.state[j], self.state[i]
        reward = self._get_reward(i, j)
        terminated = self._is_terminated()
        truncated = self._ticks > self._max_steps
        self._ticks += 1
        return self.state.copy(), reward, terminated, truncated, dict()


    def _is_terminated(self) -> bool:
        return np.array_equal(self.state[:self.n], self.GOAL[:self.n])


    def _get_reward(self, i: int, j: int) -> float:
        if i == j:
            return -0.25
        tau = self._get_tau()
        delta = tau - self._prev_tau
        reward = delta * 1.0
        reward -= 0.005 # Time penalty
        if self._is_terminated():
            reward += 10.0
        self._prev_tau = tau
        return float(reward)


    def set_n(self, new_n: int) -> None:
        self.n = min(new_n, self.max_n)


    def action_masks(self) -> np.ndarray:
        masks_i = np.zeros(self.max_n, dtype=bool)
        masks_i[:self.n] = True
        return np.concatenate([masks_i, masks_i])
