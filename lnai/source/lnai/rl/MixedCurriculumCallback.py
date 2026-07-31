# MixedCurriculumCallback.py

import numpy as np
from typing import Callable
from stable_baselines3.common.callbacks import BaseCallback

class MixedCurriculumCallback(BaseCallback):
    def __init__(self, target_success_rate: float = 0.80, base_window_size: int = 80, verbose: int = 1, window_update: Callable | None=None):
        super().__init__(verbose)
        self.target_success_rate = target_success_rate
        self.base_window_size = base_window_size
        self.successes = []
        self.current_stage = 2
        self.starting_stage = 2
        self._window_update = window_update if window_update is not None else lambda n: n ** 2
        self._update_window_size()

    def _update_window_size(self) -> None:
        self.window_size = self.base_window_size * self._window_update(self.current_stage / self.starting_stage)
    

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is not None and infos is not None:
            for idx, done in enumerate(dones):
                if done:
                    info = infos[idx]
                    if "is_success" in info:
                        self.successes.append(float(info["is_success"]))

                    if len(self.successes) > self.window_size:
                        self.successes.pop(0)

        if len(self.successes) >= self.window_size:
            current_sr = np.mean(self.successes)
            if current_sr >= self.target_success_rate:# and self.current_stage < 10:
                self.current_stage += 1
                self.successes.clear()

                self.training_env.env_method("set_max_stage", self.current_stage)
                self.training_env.reset()

                self._update_window_size()

                if self.verbose > 0:
                    print(
                        f"\n[CURRICULUM ADVANCED] Advanced Max Stage to N = {self.current_stage} "
                        f"(SR: {current_sr * 100:.1f}%)\n"
                    )

        return True
