# Experts.py

import numpy as np

class BubbleSortExpert:
    def __init__(self, pair_to_action: dict | None=None):
        self.pair_to_action = pair_to_action

    def get_action(self, obs: np.ndarray, current_n: int) -> int | tuple[int, int]:
        # state = obs[:-1]
        state = obs[:current_n]
        # Return first adjacent out-of-order pair
        for i in range(current_n - 1):
            if state[i] > state[i + 1]:
                pair = (i, i + 1)
                if self.pair_to_action is not None:
                    return self.pair_to_action[pair]
                return pair
        return 0


import numpy as np

class QuickSortExpert:
    """
    Expert coach using QuickSort with Hoare's partitioning scheme.
    Greedily computes and returns the next required pairwise swap action
    for the given state observation.
    """
    def __init__(self, pair_to_action: dict | None=None):
        self.pair_to_action = pair_to_action

    def get_action(self, obs: np.ndarray, current_n: int) -> int | tuple[int, int]:
        state = obs[:current_n].copy()
        
        # Explicit stack to simulate recursive partitions: (low, high)
        stack = [(0, current_n - 1)]

        while stack:
            low, high = stack.pop()
            if low >= high:
                continue

            # Hoare Partition Scheme
            pivot = state[low]  # Choose initial element as pivot
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
                    # Partition completed for this sub-range; push child partitions.
                    # Push right partition first so left partition is processed next (LIFO stack)
                    stack.append((j + 1, high))
                    stack.append((low, j))
                    break

                # Found an out-of-order pair relative to pivot
                if state[i] != state[j]:
                    # Map indices to (min, max) key used in pair_to_action dictionary
                    pair = (min(i, j), max(i, j))
                    if self.pair_to_action is not None:
                        return self.pair_to_action.get(pair, 0)
                    return pair
                else:
                    # Advance pointers locally if elements are duplicate values
                    state[i], state[j] = state[j], state[i]

        return 0  # Return action 0 if array is already sorted
