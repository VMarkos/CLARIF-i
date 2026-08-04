# MixedCurriculumCallback.py

from __future__ import annotations

from typing import Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MixedCurriculumCallback(BaseCallback):
    """
    Success-gated curriculum with optional return-based advancement.

    Always writes the new stage via env_method('set_max_stage', ...), which
    (after the CurriculumSortEnv refactor) also rescales episode horizons.
    """

    def __init__(
        self,
        target_success_rate: float = 0.80,
        base_window_size: int = 80,
        verbose: int = 1,
        window_update: Callable | None = None,
        reward_thresh: float | None = 10.5,
        eval_freq: int = 1000,
        max_stage: int | None = None,
    ):
        super().__init__(verbose)
        self.target_success_rate = target_success_rate
        self.base_window_size = base_window_size
        self.successes: list[float] = []
        self.returns: list[float] = []
        self.current_stage = 2
        self.starting_stage = 2
        self.reward_thresh = reward_thresh
        self.eval_freq = eval_freq
        self.max_stage_cap = max_stage
        self._window_update = (
            window_update if window_update is not None else (lambda n: n ** 2)
        )
        self._update_window_size()

    def _update_window_size(self) -> None:
        self.window_size = int(
            self.base_window_size
            * self._window_update(self.current_stage / self.starting_stage)
        )

    def _stage_cap(self) -> int:
        if self.max_stage_cap is not None:
            return self.max_stage_cap
        try:
            return int(self.training_env.get_attr("max_n")[0])
        except Exception:
            return 10**9

    def _advance(self, reason: str) -> None:
        cap = self._stage_cap()
        if self.current_stage >= cap:
            return
        self.current_stage += 1
        self.successes.clear()
        self.returns.clear()
        self.training_env.env_method("set_max_stage", self.current_stage)
        self.training_env.reset()
        self._update_window_size()
        if self.verbose > 0:
            print(
                f"\n[CURRICULUM ADVANCED] Max stage -> N = {self.current_stage} "
                f"({reason})\n"
            )

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        rewards = self.locals.get("rewards")

        if dones is not None and infos is not None:
            for idx, done in enumerate(dones):
                if done:
                    info = infos[idx]
                    if "is_success" in info:
                        self.successes.append(float(info["is_success"]))
                    # Monitor / EpisodeStats often stash episodic return in info
                    if "episode" in info and "r" in info["episode"]:
                        self.returns.append(float(info["episode"]["r"]))
                    elif rewards is not None:
                        # Fallback: not true episodic return; skip
                        pass
                    if len(self.successes) > self.window_size:
                        self.successes.pop(0)
                    if len(self.returns) > self.window_size:
                        self.returns.pop(0)

        # Primary gate: rolling success rate (existing CPAB behaviour).
        if len(self.successes) >= self.window_size:
            current_sr = float(np.mean(self.successes))
            if current_sr >= self.target_success_rate:
                self._advance(f"SR={current_sr * 100:.1f}%")
                return True

        # Secondary gate (SortingCallback-style): mean episodic return.
        if (
            self.reward_thresh is not None
            and self.n_calls % self.eval_freq == 0
            and len(self.returns) > 0
        ):
            mean_r = float(np.mean(self.returns[- min(len(self.returns), self.window_size) :]))
            if mean_r > self.reward_thresh:
                self._advance(f"mean_return={mean_r:.2f}")

        return True
