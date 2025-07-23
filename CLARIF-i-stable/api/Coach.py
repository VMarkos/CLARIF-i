# api/Coach.py

from typing import Callable

from .State import State
from .Rule import Rule

from copy import deepcopy

class Coach:
    """
    The coach maintains target rules and provides feedback to the learner.
    
    Attributes:
        target_rules: List of rules that represent the desired behavior
    """

    def __init__(self, target_rules: Callable[[State], Rule]) -> None:
        """ Initialise the coach with target rules """
        # self.target_rules: dict[State, Rule] = { rule.condition: rule for rule in sorted(target_rules, reverse=True) }
        self.target_rules = target_rules
    
    def evaluate_inference(self, start_state: State, goal_state: State,
                           traces: list[list[tuple[State, Rule | None]]]) -> tuple[bool, list[Rule]]:
        
        # print("Traces:", len(traces))
        if not traces:
            return False, self._generate_goal_rules(start_state)
        
        if goal_state in ( trace[0] for trace in traces ):
            return True, []
        
        # print("traces[-1]", traces[-1][0], [ (str(s), str(r)) for s, r in traces[-1][1] ])
        return False, self._generate_goal_rules(traces[-1][0])
        
    def _generate_goal_rules(self, current_state: State) -> list[Rule]:
        feedback_rules: list[Rule] = []
        # print(f"Current state: {current_state}")
        # print("Target rules:","\n".join(map(str, self.target_rules.keys())))
        # print(current_state in self.target_rules)
        advised_rule = self.target_rules(current_state)
        # print(f"{advised_rule}")
        advised_action = advised_rule.action
        
        # FIXME This needs to be revised for multiple rule output (e.g., partial condition states)
        # for key, goal_value in advised_action:
        #     current_value = current_state.get(key)
        #     action_state = State()
        #     if current_value != goal_value:
        #         action_state.set(key, goal_value)
        # print("Action state:", action_state)
        if advised_action:
            feedback_rules.append(advised_rule)
        
        return feedback_rules
