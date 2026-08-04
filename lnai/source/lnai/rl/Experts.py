# Experts.py
#
# Swap-based coaching experts. Actions are always factored index pairs (i, j)
# as np.ndarray([i, j]), matching MultiDiscrete action spaces.

from __future__ import annotations

import numpy as np


def _as_action(i: int, j: int) -> np.ndarray:
    return np.asarray([i, j], dtype=np.int64)


def _noop() -> np.ndarray:
    return _as_action(0, 0)


class BubbleSortExpert:
    def __init__(self, pair_to_action: dict | None = None):
        # pair_to_action kept for backward compatibility; ignored for MultiDiscrete.
        self.pair_to_action = pair_to_action

    def get_action(self, obs: np.ndarray, current_n: int) -> np.ndarray:
        state = np.asarray(obs).reshape(-1)[:current_n]
        for i in range(current_n - 1):
            if state[i] > state[i + 1]:
                return _as_action(i, i + 1)
        return _noop()


class SelectionSortExpert:
    def __init__(self, pair_to_action_map: dict | None = None):
        self.pair_to_action = pair_to_action_map

    def get_action(self, obs: np.ndarray, current_n: int) -> np.ndarray:
        arr = np.asarray(obs).reshape(-1)[:current_n]
        for i in range(current_n - 1):
            min_offset = int(np.argmin(arr[i:]))
            min_idx = i + min_offset
            if min_idx != i:
                return _as_action(i, min_idx)
        return _noop()


class QuickSortExpert:
    """
    Expert coach using QuickSort with Hoare's partitioning scheme.
    Returns the next required pairwise swap as (i, j).
    """

    def __init__(self, pair_to_action: dict | None = None):
        self.pair_to_action = pair_to_action

    def get_action(self, obs: np.ndarray, current_n: int) -> np.ndarray:
        state = np.asarray(obs).reshape(-1)[:current_n].copy()
        stack = [(0, current_n - 1)]

        while stack:
            low, high = stack.pop()
            if low >= high:
                continue

            pivot = state[low]
            i = low - 1
            j = high + 1

            while True:
                i += 1
                while state[i] < pivot:
                    i += 1

                j -= 1
                while state[j] > pivot:
                    j -= 1

                if i >= j:
                    stack.append((j + 1, high))
                    stack.append((low, j))
                    break

                if state[i] != state[j]:
                    return _as_action(i, j)
                state[i], state[j] = state[j], state[i]

        return _noop()
