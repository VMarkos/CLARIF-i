"""
Learner implementation for the coachable search framework.
"""

from typing import Dict, List, Tuple, Set, Optional

from .rule import Rule

class Learner:
    """
    The learner maintains a hypothesis and updates it based on coach feedback.
    
    Attributes:
        hypothesis: List of rules that represent the learner's current understanding
    """
    
    def __init__(self, initial_rules: List[Rule]):
        """Initialize the learner with initial rules."""
        self.hypothesis = sorted(initial_rules, reverse=True)
    
    def search_path(self, start_state: Dict[str, str], goal_state: Dict[str, str]) -> Tuple[bool, List[List[Tuple[Dict[str, str], Optional[Rule]]]]]:
        """
        Search for paths from start state to goal state using current rules.
        
        Args:
            start_state: The initial state
            goal_state: The goal state
            
        Returns:
            Tuple of (success, traces) where:
            - success is True if a path was found, False otherwise
            - traces is a list of reasoning traces, each trace being a list of (state, rule) pairs
        """
        # Initialize traces
        traces = []
        
        # Check if start state already matches goal state
        if self._states_match(start_state, goal_state):
            traces.append([(start_state, None)])
            return True, traces
        
        # Try to find paths using current rules
        visited = set()
        queue = [(start_state, [])]
        
        while queue:
            current_state, path = queue.pop(0)
            state_key = self._state_to_key(current_state)
            
            if state_key in visited:
                continue
                
            visited.add(state_key)
            
            # Check if current state matches goal state
            if self._states_match(current_state, goal_state):
                # Convert path to trace format
                trace = [(start_state, None)]
                for state, rule in path:
                    trace.append((state, rule))
                traces.append(trace)
                continue
            
            # Try to apply rules to current state
            for rule in self.hypothesis:
                if self._rule_applies(rule, current_state):
                    new_state = self._apply_rule(rule, current_state)
                    new_path = path + [(new_state, rule)]
                    queue.append((new_state, new_path))
        
        # If we found any traces, return success
        if traces:
            return True, traces
        
        # If no complete paths found, return partial paths
        partial_traces = []
        for state, path in queue:
            trace = [(start_state, None)]
            for s, r in path:
                trace.append((s, r))
            partial_traces.append(trace)
        
        return False, partial_traces
    
    def _states_match(self, state1: Dict[str, str], state2: Dict[str, str]) -> bool:
        """Check if two states match."""
        return all(state1.get(k) == v for k, v in state2.items())
    
    def _state_to_key(self, state: Dict[str, str]) -> str:
        """Convert state to a string key for visited set."""
        return ','.join(f"{k}={v}" for k, v in sorted(state.items()))
    
    def _rule_applies(self, rule: Rule, state: Dict[str, str]) -> bool:
        """Check if a rule can be applied to a state."""
        return all(state.get(k) == v for k, v in rule.condition.items())
    
    def _apply_rule(self, rule: Rule, state: Dict[str, str]) -> Dict[str, str]:
        """Apply a rule to a state."""
        new_state = state.copy()
        new_state.update(rule.preference)
        return new_state
    
    def update_hypothesis(self, feedback_rules: List[Rule]):
        """
        Update the learner's hypothesis based on feedback.
        
        Args:
            feedback_rules: List of rules provided as feedback
        """
        # Add new rules to hypothesis
        for rule in feedback_rules:
            if rule not in self.hypothesis:
                self.hypothesis.append(rule)
        
        # Sort rules by priority
        self.hypothesis.sort(reverse=True)
        
        # Remove duplicate rules (keeping highest priority)
        seen_conditions = set()
        unique_rules = []
        for rule in self.hypothesis:
            condition_key = tuple(sorted(rule.condition.items()))
            if condition_key not in seen_conditions:
                seen_conditions.add(condition_key)
                unique_rules.append(rule)
        self.hypothesis = unique_rules 