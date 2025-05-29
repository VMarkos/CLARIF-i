# api/State.py

from typing import Any

class State:
    def __init__(self, state: dict[str, Any] = dict()) -> None:
        self.state: dict[str, Any] = state

    def update(self, other: "State") -> "State":
        self.state.update(other.state)

    def get(self, key: str) -> Any:
        if key not in self.state.keys():
            raise KeyError(f"Variable '{key}' not found!")
        return self.state.get(key)
    
    def set(self, key: str, val: Any) -> None:
        self.state[key] = val
    
    def swap(self, k1: str, k2: str) -> None:
        temp: str = self.state.get(k1)
        self.state[k1] = self.state.get(k2)
        self.state[k2] = temp

    def __bool__(self) -> bool:
        return len(self.state) != 0

    def __iter__(self) -> iter:
        return iter(self.state.items()) # FIXME Maybe `iter()` is not needed here

    def __deepcopy__(self, memo) -> "State":
        copycat: "State" = State(self.state.copy())
        return copycat

    def __le__(self, other: "State") -> bool:
        if not isinstance(other, State):
            return False
        try:
            return all(other.state[k] == v for k, v in self.state.items())
        except KeyError:
            return False
        else:
            raise e

    def __hash__(self) -> int:
        """Compute state hash by its (unique) string representation"""
        h = hash(str(self))
        # print(self, h)
        return h

    def __str__(self) -> str:
        """String representation of state as a dictionary"""
        return ','.join(f"{k}={v}" for k, v in sorted(self.state.items()))

    def __eq__(self, other: "State") -> bool:
        """Boolean equality based on dictionary equality"""
        if not isinstance(other, State):
            return False
        if len(other.state.keys()) != len(self.state.keys()):
            return False
        try:
            keys_match = all(self.state[k] == v for k, v in other)
        except KeyError:
            return False
        return keys_match

    def __len__(self) -> int:
        return len(self.state)

    @classmethod
    def from_str(cls, tc_str: str) -> "State":
        kv_dict = { x[0]: int(x[1]) for x in map(lambda s: s.split("="), tc_str.split(",")) }
        return cls(kv_dict)
