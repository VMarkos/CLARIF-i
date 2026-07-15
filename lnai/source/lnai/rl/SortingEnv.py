# rl/Environment.py

import numpy as np
import cv2
from gymnasium import Env, spaces
from matplotlib import pyplot as plt
from lnai.api.State import State
from lnai.api.Action import Action

import random

# Should this be elsewhere?
random.seed(5679813004)


class SortingEnv(Env):
    def __init__(self, n: int, target_steps: int | None=None) -> None:
        super(SortingEnv, self).__init__()
        
        # Instance "globals"
        self.render_mode = 'ansi'

        # GOAL
        self.GOAL = State(dict(zip(range(n), range(n))))

        # Initialize object fields
        self.n = n
        self._ticks = 0
        if target_steps is None:
            self._target_steps = self.n * np.log(n)
        else:
            self._target_steps = target_steps

        # Discrete obesrvation space containing n^n (not all valid) different objects
        self.observation_space = spaces.MultiDiscrete([n] * n)

        # Action space where each action is a swap
        self.action_space = spaces.Discrete(n * (n - 1) // 2)

        # Initialize state
        self.__init_state()


    def __init_state(self) -> None:
        """Initialize state to a random state"""
        _rand_state_dict = dict(zip(range(self.n), random.sample(range(self.n), k=self.n)))
        self.state: State = State(_rand_state_dict)
        self._previous_tau = self.state.kendall_tau(self.GOAL)
        self._ticks = 0



    def reset(self, seed=None, options=None) -> tuple[State, dict]:
        """Resets internal state"""
        super().reset(seed=seed, options=options)
        self.__init_state()
        return self.state.as_ndarray(), dict()


    def render(self) -> str | None:
        """Renders the environment as ANSI string."""
        print(self.state)
        return str(self.state)


    def step(self, action: int) -> tuple:
        """Makes a step forward by applying an action to the current setting"""
        swap = self.__get_swap(action)
        self.state.swap(*swap)
        reward = self.__get_reward()
        terminated = self.state == self.GOAL
        truncated = False
        self._ticks += 1
        return self.state.as_ndarray(), reward, terminated, truncated, dict()


    def __get_reward(self) -> float:
        tau = self.state.kendall_tau(self.GOAL)
        reward = tau - 1.0 * self._ticks / self._target_steps
        if self.state == self.GOAL:
            reward += 10.0
        self._previous_tau = tau
        return reward


    def __get_swap(self, i: int) -> tuple[int]:
        k = 0
        for a in range(self.n - 1):
            for b in range(a + 1, self.n):
                if k == i:
                    return (a, b)
                k += 1


    def show(self) -> None:
        """Shows the internal state on canvas - thin wrapper around plt.imshow()"""
        plt.imshow(self.canvas)
        plt.show()
