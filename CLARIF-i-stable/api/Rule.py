# api/Rule.py

"""
Rule representation for the coachable search framework.
"""

from .State import State
from .Action import Action
from .Predicate import Predicate
from copy import deepcopy

class Rule:
    def __init__(self, name: str, condition: State | Predicate, action: Action, priority: int = 1, explanation: str = "") -> None:
        self.name: str = name
        self.condition: State | Predicate = condition
        self.action: Action = action
        self.priority: int = priority
        self.explanation: str = explanation
        self._relational: bool = isinstance(self.condition, Predicate)

    def __key(self) -> tuple:
        """Compute rule key by body and head"""
        return (self.condition, self.action)

    def __hash__(self) -> int:
        """Compute rule hash by body and head"""
        return hash(self.__key())
    
    def __eq__(self, other: "Rule") -> bool:
        """Boolean equality based on body and head"""
        if not isinstance(other, Rule):
            return False
        # return self.condition == other.condition # temporary change
        return self.__key() == other.__key()

    def __str__(self):
        return f"{self.name}: IF {self.condition} THEN {self.action} (priority: {self.priority})"

    def __repr__(self):
        return self.__str__()

    # TODO Reconsider how this can be implemented using `Action`
    # def to_dict(self) -> dict:
    #     """Convert the rule to a dictionary for serialization."""
    #     return {
    #         'name': self.name,
    #         'condition': self.condition.__dict__,
    #         'action': self.action.__dict__,
    #         'priority': self.priority
    #     }

    def applies(self, state: State) -> bool:
        """Checks if a rule can be applied to a state, which is whenever the rule's body is covered by `state`."""
        if self._relational:
            return self.condition.evaluate(state)
        return self.condition <= state
    
    def apply(self, state: State) -> State:
        """ Applies a rule to a state by updating the state with the action (head) of the rule. """
        new_state = deepcopy(state)
        new_state.update(self.action.apply(new_state))
        return new_state

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
