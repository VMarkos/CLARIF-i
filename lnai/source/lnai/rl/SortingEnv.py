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
    def __init__(self, n: int, action_generator) -> None:
        super(SortingEnv, self).__init__()
        
        # Instance "globals"
        self.HEIGHT = 600
        self.WIDTH = 800

        # Initialize object fields
        self.n = n

        # Discrete obesrvation space containing n different objects
        self.observation_space = spaces.Discrete(n)

        # Action space where each action is a swap
        self.action_space = spaces.Discrete(n * (n - 1) // 2)

        # Initialize state
        self.__init_state()

        # Initialize canvas
        self._canvas_shape = (self.HEIGHT, self.WIDTH, 3)
        self.canvas = np.ones(self._canvas_shape) * 255


    def __init_state(self) -> None:
        """Initialize state to a random state"""
        _rand_state_dict = dict(zip(range(self.n), random.sample(range(self.n), k=self.n)))
        self.state: State = State(_rand_state_dict)

        


    def draw_state_on_canvas(self) -> None:
        """Draws the current state as a string on the environment canvas"""
        self.canvas = np.ones(self._canvas_shape).astype(np.uint8) * 255
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(self.state)
        (retval, text_w), text_h = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.LINE_AA,
            0
        )
        text_pos = (self.WIDTH // 2 - text_w // 2, self.HEIGHT // 2 + text_h // 2) # FIXME: Need to subtract text dimensions
        text_color = (0, 0, 0)
        self.canvas = cv2.putText(
            self.canvas,
            text,
            text_pos,
            text_font,
            1.0,
            text_color,
            1,
            cv2.LINE_AA
        )


    def reset(self) -> None:
        """Resets internal state and redraws canvas"""
        self.__init_state()
        self.draw_state_on_canvas()


    def step(self, action: int) -> None:
        """Makes a step forward by applying an action to the current setting"""
        self.state = action.apply(self.state)


    def __get_swap(self, i: int) -> Action:
        k = 0
        for a in range(n - 1):
            for b in range(a + 1, n):
                if k == i:
                    return (a, b)


    def show(self) -> None:
        """Shows the internal state on canvas - thin wrapper around plt.imshow()"""
        plt.imshow(self.canvas)
        plt.show()
