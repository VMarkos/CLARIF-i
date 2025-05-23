from .State import State
from typing import Callable

class Action:
    def __init__(self, callback: Callable[[State], State] | None = None, name: str = "No action") -> None:
        self.callback: Callable[[State], State] = callback
        self.name: str = name

    def apply(self, state: State) -> State:
        """Assuming that `self.callback` does not mutate `state`."""
        if self.callback == None:
            return state
        return self.callback(state)

    def __key(self) -> int:
        return (self.callback)

    def __hash__(self) -> int:
        return hash(self.__key())
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Action):
            return False
        return self.__key() == other.__key()
   
    def __bool__(self) -> bool:
        return self.callback != None

    def __str__(self) -> str:
        return self.name
