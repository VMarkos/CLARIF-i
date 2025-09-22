from .State import State
from typing import Callable


class Action:
    def __init__(
        self,
        action_fn: Callable[State, State] | None = None,
        name: str = "No action",
        callback: Callable | None = None,
    ) -> None:
        self.action_fn: Callable[State, State] = action_fn
        self.name: str = name
        self.callback = callback

    def apply(self, state: State) -> State:
        """Assuming that `self.callback` does not mutate `state`."""
        if self.action_fn == None:
            return state
        post_action_state = self.action_fn(state)
        if self.callback == None:
            return post_action_state
        self.callback(post_action_state)
        return post_action_state

    def __key(self) -> int:
        return self.name

    def __hash__(self) -> int:
        return hash(self.__key())

    def __eq__(self, other) -> bool:
        if not isinstance(other, Action):
            return False
        return self.__key() == other.__key()

    def __bool__(self) -> bool:
        return self.action_fn != None

    def __str__(self) -> str:
        return self.name
