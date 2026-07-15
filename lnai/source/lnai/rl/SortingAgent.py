# rl/SortingAgent.py
#
# RL sorting agent, which tries to learn how to sort lists of integers.


import numpy as np
from tqdm import tqdm
from lnai.rl.SortingEnv import SortingEnv
from lnai.rl.QLearner import QLearner
from lnai.api.State import State


class SortingAgent(QLearner):
    def __init__(
            self,
            n: int,
            n_episodes: int,
            env: SortingEnv,
            q_hyperparams: dict,
        ) -> None:
        super().__init__(**q_hyperparams)
        self.n = n
        self.n_episodes = n_episodes
        self.env = env


    def get_action(self, obs: State) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return super().get_action(obs)



    def learn(self) -> None:
        for _ in tqdm(range(self.n_episodes)):
            state, info = self.env.reset()
            terminated, truncated = False, False
            while not terminated and not truncated:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.update(state, action, reward, terminated, next_state)
                state = next_state
            self.reduce_eps() 
