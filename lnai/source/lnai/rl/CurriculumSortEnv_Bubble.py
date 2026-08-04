"""Bubble-favouring curriculum env: adjacent swaps only via masking + hard reject."""

from __future__ import annotations

import numpy as np

from lnai.rl.CurriculumSortEnv import CurriculumSortEnv


class CurriculumSortEnv_Bubble(CurriculumSortEnv):
    """Same as CurriculumSortEnv, but only adjacent (i, i+1) / (i+1, i) swaps are legal."""

    def action_masks(self) -> np.ndarray:
        """
        Factorised masks cannot encode adjacency exactly; we allow active indices
        and enforce adjacency in step(). Leading (current_n - 1) positions are the
        natural bubble pivots, so we still zero inactive slots.
        """
        return super().action_masks()

    def step(self, action):
        i = int(action[0])
        j = int(action[1])
        if abs(i - j) != 1 and not (i == j):
            self.steps += 1
            return self.state.copy(), -2.0, True, False, {"is_success": False}
        # i == j still handled by parent (noop penalty); adjacent handled normally.
        return super().step(action)
