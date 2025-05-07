# api/State.py

class State:
    def __init__(self, state: dict[str, str] = dict()) -> None:
        self.state: dict[str, str] = state

    def update(self, other: "State") -> "State":
        self.state.update(other.state)

    def get(self, key: str) -> str:
        if key not in self.state.keys():
            raise KeyError(f"Variable '{key}' not found!")
        return self.state.get(key)
    
    def set(self, key: str, val: str) -> None:
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
        return all(self.state[k] == v for k, v in other.state.items())

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
        # print("Checking for eq")
        if not isinstance(other, State):
            return False
        if len(other.state.keys()) != len(self.state.keys()):
            return False
        try:
            keys_match = all(self.state[k] == v for k, v in other)
        except KeyError:
            return False
        # print(self.state, other.state)
        return keys_match