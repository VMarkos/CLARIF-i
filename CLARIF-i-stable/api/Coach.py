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
        self.target_rules: dict[State, Rule] = { rule.condition: rule for rule in sorted(target_rules, reverse=True) }
    
    def evaluate_inference(self, start_state: State, goal_state: State,
                           traces: list[list[tuple[State, Rule | None]]]) -> tuple[bool, list[Rule]]:
        
        # print("Traces:", traces)
        if not traces:
            return False, self._generate_goal_rules(start_state)
        
        if goal_state in ( trace[-1] for trace in traces ):
            return True, []
        
        print("traces[-1]", [ x for x in map(str, traces[-1]) ])
        return False, self._generate_goal_rules(traces[-1][0])
        
    # TODO Recall that for full states, priorities do not actually matter - just for partial states

    def _generate_goal_rules(self, current_state: State) -> list[Rule]:
        feedback_rules: list[Rule] = []
        print(f"Current state: {current_state if type(current_state) == State else current_state[0]}")
        # print("Target rules:","\n".join(map(str, self.target_rules.keys())))
        # print(current_state in self.target_rules)
        advised_rule = self.target_rules[current_state]
        advised_action = advised_rule.action

        for key, goal_value in advised_action:
            current_value = current_state.get(key)
            action_state = State()
            if current_value != goal_value:
                action_state.set(key, goal_value)
        print("Action state:", action_state)
        if action_state:
            rule = Rule(
                name="",
                condition=deepcopy(current_state),
                action=action_state,
                priority=1,
                explanation=f"Change {key} from {current_value} to {goal_value}"
            )
            feedback_rules.append(rule)
        
        return feedback_rules