"""
Rule representation for the coachable search framework.
"""

from dataclasses import dataclass
from typing import Dict

class Rule:
    def __init__(self, name: str, condition: dict, action: dict, priority: int = 1):
        self.name = name
        self.condition = condition
        self.action = action
        self.priority = priority

    def __str__(self):
        return f"{self.name}: IF {self.condition} THEN {self.action} (priority: {self.priority})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        """Convert the rule to a dictionary for serialization."""
        return {
            'name': self.name,
            'condition': self.condition,
            'action': self.action,
            'priority': self.priority
        }

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