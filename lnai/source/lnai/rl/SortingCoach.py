# rl/SortingCoach.py
#
# Class implementing a generic Algorithm-Based Sorting Coach

from numpy import ndarray, flatnonzero, array
from typing import Callable
from utils import monitor_sequences
from copy import deepcopy


class SortingCoach:
    def __init__(self, algorithm: Callable[[ndarray], ndarray]) -> None:
        self.__log = dict()
        __decorator = monitor_sequences(self.__log)
        self.algorithm = __decorator(algorithm)
        self.__swaps: dict[tuple, tuple] = dict()


    def update_state(self, state: ndarray) -> None:
        '''Records all steps executed by the sorting algorithm on a given array and updates self.__swaps'''
        sorted_state = self.algorithm(deepcopy(state))
        intermediate_states = next(iter(self.__log.values()))[::2] # Assuming just one key is present
        for prev_state, next_state in zip(intermediate_states[:-1], intermediate_states[1:]):
            swap = tuple(int(x) for x in flatnonzero(next_state - prev_state))
            if swap == ():
                continue
            if len(swap) != 2:
                raise ValueError(f'Expected 2 indices to swap, not {",".join(map(str, swap))}')
            self.__swaps[tuple(prev_state.flatten())] = swap


    def get_advice(self, state: ndarray) -> tuple | None:
        '''Returns a tuple with the suggested swap or None in case no such state should have been encountered.'''
        return self.__swaps.get(tuple(state.flatten()), None)


    def get_state_at(self, state: ndarray, k: int) -> ndarray:
        sorted_state = self.algorithm(deepcopy(state))
        intermediate_states = next(iter(self.__log.values()))[::2]
        k = min(k, len(intermediate_states) - 1)
        return intermediate_states[-k-1]



