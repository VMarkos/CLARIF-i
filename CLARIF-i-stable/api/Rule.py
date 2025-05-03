# api/Rule.py

"""
Rule representation for the coachable search framework.
"""

from State import State
from copy import deepcopy

class Rule:
    def __init__(self, name: str, condition: State, action: State, priority: int = 1):
        self.name: str = name
        self.condition: list[State] = condition
        self.action: State = action
        self.priority: int = priority

    def __str__(self):
        return f"{self.name}: IF {self.condition} THEN {self.action} (priority: {self.priority})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        """Convert the rule to a dictionary for serialization."""
        return {
            'name': self.name,
            'condition': self.condition.__dict__,
            'action': self.action,
            'priority': self.priority
        }

    def applies(self, state: State) -> bool:
        """Checks if a rule can be applied to a state, which is whenever the rule's body is covered by the `state`."""
        return self.condition <= state
    
    def apply(self, state: State) -> State:
        new_state = deepcopy(state)
        # new_state.

    @classmethod
    def from_dict(cls, data: dict) -> 'Rule':
        """Create a Rule instance from a dictionary."""
        return cls(
            name=data['name'],
            condition=data['condition'],
            action=data['action'],
            priority=data['priority']
        )

    def __lt__(self, other: "Rule"):
        """Compare rules based on priority."""
        return self.priority < other.priority