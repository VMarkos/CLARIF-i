# rl/Environment.py

import numpy as np
from gymnasium import Env, spaces
from matplotlib import pyplot as plt
from lnai.api.State import State

import random

# Should this be elsewhere?
# random.seed(5679813004)


class SortingEnv(Env):
    def __init__(self, max_n: int, start_size: int=4, target_steps: int | None=None) -> None:
        super(SortingEnv, self).__init__()
        
        # Instance "globals"
        self.render_mode = 'ansi'

        # GOAL
        self.GOAL = State(dict(zip(range(n), range(n))))

        # Initialize object fields
        self.n = start_size
        self.max_n = max_n
        self._ticks = 0
        if target_steps is None:
            self._target_steps = self.n * np.log(n)
        else:
            self._target_steps = target_steps

        # Discrete obesrvation space containing n^n (not all valid) different objects
        self.observation_space = spaces.MultiDiscrete([n] * n)

        # Action space where each action is a swap
        self.action_space = spaces.MultiDiscrete([n, n])

        # Initialize state
        self.__init_state()


    def __init_state(self) -> None:
        """Initialize state to a random state"""
        _rand_state_dict = dict(zip(range(self.n), random.sample(range(self.n), k=self.n)))
        self.state: State = State(_rand_state_dict)
        self._ticks = 0
        self._prev_inv = self.state.inversions_ratio()



    def reset(self, seed=None, options=None) -> tuple[State, dict]:
        """Resets internal state"""
        super().reset(seed=seed, options=options)
        self.__init_state()
        return self.state.as_ndarray(), dict()


    def render(self) -> str | None:
        """Renders the environment as ANSI string."""
        print(self.state)
        return str(self.state)


    def step(self, action) -> tuple:
        """Makes a step forward by applying an action to the current setting"""
        action = np.array(action).flatten()
        self.state.swap(*action)
        reward = self.__get_reward(action)
        terminated = self.state == self.GOAL
        truncated = False # self._ticks > 4_000
        self._ticks += 1
        return self.state.as_ndarray(), reward, terminated, truncated, dict()


    def __get_reward(self, action) -> float:
        if action[0] == action[1]:
            return -0.5
        # tau = self.state.kendall_tau(self.GOAL)
        inv = self.state.inversions_ratio()
        delta = inv - self._prev_inv
        reward = 2.0 * delta - 0.1 * self._ticks / self._target_steps
        if self.state == self.GOAL:
            reward += 10.0
        self._prev_inv = inv
        return reward


    def set_n(self, new_n: int) -> None:
        self.n = min(new_n, self.max_n)


    def action_masks(self) -> np.ndarray:
        masks_i = np.zeros(self.max_n, dtype=bool)
        masks_i[:self.max_n] = True
        return np.concatenate([masks_i, masks_i])
