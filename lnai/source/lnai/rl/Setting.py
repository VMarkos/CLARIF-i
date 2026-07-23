# rl/Setting.py
#
# Wrapper around training and testing Settings

from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from lnai.rl.SortingEnv import SortingEnv
from lnai.rl.SortingCallback import SortingCallback


class GenericSetting:
    def __init__(self, n: int, n_timesteps: int, load: bool=Falsemodel_path: str='') -> None:
        self.n = n
        self.n_timesteps = n_timesteps
        self.env = SortingEnv(n)
        self.env = RecordEpisodeStatistics(env, n_timesteps)
        callback = SortingCallback(end_n=n, verbose=1)
        if load:
            self.model = PPO.load(
