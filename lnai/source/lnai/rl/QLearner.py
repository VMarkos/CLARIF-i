# rl/QLearner.py
#
# RL Qlearning agent, using naive table-based qlearning values.

from collections import defaultdict
import numpy as np

class QLearner:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
        n_actions: int,
    ) -> None:
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.q_values = defaultdict(lambda: np.zeros(n_actions))
        self.training_error = []


    def get_action(self, obs: object) -> int:
        return int(np.argmax((self.q_values[obs])))


    def update(
        self,
        obs: object,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: object,
    ) -> None:
        future_q = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q
        t_diff = target - self.q_values[obs][action]
        self.q_values[obs][action] = self.q_values[obs][action] + self.learning_rate * t_diff
        self.training_error.append(t_diff)


    def reduce_eps(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
