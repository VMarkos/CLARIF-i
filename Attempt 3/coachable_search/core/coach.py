"""
Coach implementation for the coachable search framework.
"""

from typing import Dict, List, Tuple, Optional

from .rule import Rule

class Coach:
    """
    The coach maintains target rules and provides feedback to the learner.
    
    Attributes:
        target_rules: List of rules that represent the desired behavior
    """
    
    def __init__(self, target_rules: List[Rule]):
        """Initialize the coach with target rules."""
        self.target_rules = sorted(target_rules, reverse=True)
    
    def evaluate_inference(self, start_state: Dict[str, str], goal_state: Dict[str, str],
                         traces: List[List[Tuple[Dict[str, str], Optional[Rule]]]]) -> Tuple[bool, List[Rule]]:
        """
        Evaluate the learner's paths and provide feedback.
        
        Args:
            start_state: The initial state
            goal_state: The goal state
            traces: List of reasoning traces
            
        Returns:
            Tuple of (is_correct, feedback_rules) where:
            - is_correct is True if the paths are correct, False otherwise
            - feedback_rules is a list of rules to add to the learner's hypothesis
        """
        # If no traces were found, provide feedback to help reach the goal
        if not traces:
            return False, self._generate_goal_rules(start_state, goal_state)
        
        # Check if any trace reaches the goal
        for trace in traces:
            last_state, _ = trace[-1]
            if self._states_match(last_state, goal_state):
                return True, []
        
        # If no trace reaches the goal, provide feedback
        return False, self._generate_goal_rules(traces[-1][-1][0], goal_state)
    
    def _states_match(self, state1: Dict[str, str], state2: Dict[str, str]) -> bool:
        """Check if two states match."""
        return all(state1.get(k) == v for k, v in state2.items())
    
    def _generate_goal_rules(self, current_state: Dict[str, str], goal_state: Dict[str, str]) -> List[Rule]:
        """
        Generate rules to help reach the goal state.
        
        Args:
            current_state: The current state
            goal_state: The goal state
            
        Returns:
            List of rules to help reach the goal
        """
        feedback_rules = []
        
        # Find variables that need to change
        for var, goal_value in goal_state.items():
            current_value = current_state.get(var)
            if current_value != goal_value:
                # Create a rule to change this variable
                rule = Rule(
                    condition=current_state.copy(),
                    preference={var: goal_value},
                    priority=1,
                    explanation=f"Change {var} from {current_value} to {goal_value}"
                )
                feedback_rules.append(rule)
        
        return feedback_rules 