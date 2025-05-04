# api/State.py

class State:
    def __init__(self, state: dict[str, str]) -> None:
        self.state: dict[str, str] = state

    def update(self, other: "State") -> "State":
        self.state.update(other.state)

    def get(self, key: str) -> str:
        if key not in self.state.keys():
            raise KeyError(f"Variable '{key}' not found!")
        return self.state.get(key)

    def __iter__(self) -> iter:
        return iter(self.state.items()) # FIXME Maybe `iter()` is not needed here

    def __deepcopy__(self) -> "State":
        copycat: "State" = State(self.state.copy())
        return copycat

    def __le__(self, other: "State") -> bool:
        return all(self.state[k] == v for k, v in other.state.items())

    def __hash__(self) -> int:
        """Compute state hash by its (unique) string representation"""
        return hash(self.__str__())

    def __str__(self) -> str:
        """String representation of state as a dictionary"""
        return ','.join(f"{k}={v}" for k, v in sorted(self.state.items()))

    def __eq__(self, other: "State") -> bool:
        """Boolean equality based on dictionary equality"""
        if not isinstance(other, State):
            return False
        return len(other.state.keys()) == len(self.state.keys()) and all(self.state[k] == v for k, v in other.state.items())