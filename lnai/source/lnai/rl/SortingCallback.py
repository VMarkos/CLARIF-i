# SortingCallback.py
#
# stable baselines3 callback utility for sorting curriculum learning


import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class SortingCallback(BaseCallback):
    def __init__(self, start_n: int=4, end_n: int=20, eval_freq: int=10_000, reward_thresh: float=10.5, verbose=0) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.reward_thresh = reward_thresh
        self.stage_count = 0
        self.stages = [x for x in range(start_n, end_n + 1)]


    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                m_r = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                if m_r > self.reward_thresh and self.stage_count < len(self.stages) - 1:
                    self.stage_count += 1
                    new_n = self.stages[self.stage_count]
                    self.training_eng.env_method('set_n', new_n)
                    if self.verbose > 0:
                        print(f'Advancing to stage n={new_n}.')
        return True
