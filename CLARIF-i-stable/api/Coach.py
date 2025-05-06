# api/Coach.py

from .State import State
from .Rule import Rule

from copy import deepcopy

class Coach:
    """
    The coach maintains target rules and provides feedback to the learner.
    
    Attributes:
        target_rules: List of rules that represent the desired behavior
    """

    def __init__(self, target_rules: list[Rule]) -> None:
        """ Initialise the coach with target rules """
        self.target_rules: list[Rule] = sorted(target_rules, reverse=True)
    
    def evaluate_inference(self, start_state: State, goal_state: State,
                           traces: list[list[tuple[State, Rule | None]]]) -> tuple[bool, list[Rule]]:
        if not traces:
            return False, self._generate_goal_rules(start_state, goal_state)
        
        if goal_state in ( trace[-1] for trace in traces ):
            return True, []
        
        return False, self._generate_goal_rules(traces[-1], goal_state)
        
    def _generate_goal_rules(self, current_state: State, goal_state: State) -> list[Rule]:
        feedback_rules: list[Rule] = []

        for key, goal_value in goal_state:
            current_value = current_state
            if current_value != goal_value:
                rule = Rule(
                    condition=deepcopy(current_state),
                    action=State({key, goal_value}),
                    priority=1,
                    explanation=f"Change {key} from {current_value} to {goal_value}"
                )
                feedback_rules.append(rule)
        
        return feedback_rules