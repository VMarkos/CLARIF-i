# api/Predicate.py

from typing import Callable
from .State import State

class Predicate:
    def __init__(self, callback: Callable, *args, name: str = "predicate") -> None:
        self.callback = callback
        self.args = args
        self.name = name

    def evaluate(self, state: State) -> bool:
        return self.callback(state, *self.args)
