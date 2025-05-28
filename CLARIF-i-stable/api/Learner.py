# api/Learner.py

"""
Learner implementation for the coachable search framework.
"""

from functools import reduce

from typing import Dict, List, Tuple, Set, Optional

from .Rule import Rule
from .State import State

class Learner:
    """
    The learner maintains a hypothesis and updates it based on coach feedback.
    
    Attributes:
        hypothesis: List of rules that represent the learner's current understanding
    """
    
    def __init__(self, initial_rules: List[Rule] = []):
        """Initialize the learner with initial rules."""
        self.hypothesis: list[Rule] = sorted(initial_rules, reverse=True)
        self._trace: list[State] = [] # list of traces in the form of States the learner passes through
    
    def search_path(self, start_state: Dict[str, str], goal_state: Dict[str, str]) -> Tuple[bool, List[List[Tuple[State, Optional[Rule]]]]]:
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
        # Check if start state already matches goal state
        self._trace = [start_state]

        if start_state == goal_state:
            # If they match, return the start state as a trace
            # with no rules applied
            traces = [(start_state, None)]
            return True, traces
        
        # Try to find paths using current rules
        visited = set()
        queue = [(start_state, [])]
        partial_traces_dict = { start_state: set() }
        
        while queue:
            current_state, path = queue.pop(0)
            # state_key = self._state_to_key(current_state)
            
            if current_state in visited:
                continue
                
            visited.add(current_state)
            self._trace.append(current_state)
            # Check if current state matches goal state
            if current_state == goal_state:
                if current_state in partial_traces_dict:
                    partial_traces_dict[current_state].add( tuple(new_path) )
                else:
                    partial_traces_dict[current_state] = { tuple(new_path) }
                continue
            
            # Try to apply rules to current state
            top_rule = self._find_top_rule(current_state)
            if top_rule != None:
                new_state = top_rule.apply(current_state)
                new_path = path + [(new_state, top_rule)]
                queue.append((new_state, new_path))
                if new_state in partial_traces_dict:
                    partial_traces_dict[new_state].add( tuple(new_path) )
                else:
                    partial_traces_dict[new_state] = { tuple(new_path) }
        
        # If we found any traces, return success
        traces = [ (state, path) for state, paths in partial_traces_dict.items() for path in paths ]
        if traces:
            # print("RETURNING FULL TRACES")
            # print(f"\tLEARNER TRACES{[str(t[0]) for t in traces]}")
            return True, traces
        
        # print("RETURNING PARTIAL TRACES")  
        return False, traces
   
    def _find_top_rule(self, state: State) -> Rule | None:
        try:
            return reduce(lambda x, y: x if x.priority > y.priority else y, filter(lambda r: r.applies(state), self.hypothesis))
        except TypeError:
            return None

    def update_hypothesis(self, feedback_rules: List[Rule]):
        """
        Update the learner's hypothesis based on feedback.
        
        Args:
            feedback_rules: List of rules provided as feedback
        """
        # print("feedback_rules:", feedback_rules)
        # Add new rules to hypothesis
        for rule in feedback_rules:
            if rule not in self.hypothesis:
                self.hypothesis.append(rule)
        
        # Sort rules by priority
        self.hypothesis.sort(reverse=True)
        
        # Remove duplicate rules (keeping highest priority)
        # TODO : Implement a more efficient way to remove duplicates
        seen_conditions = set()
        unique_rules = []
        for rule in self.hypothesis:
            condition = rule.condition
            if condition not in seen_conditions:
                seen_conditions.add(condition)
                unique_rules.append(rule)
        # print("unique_rules", unique_rules)
        self.hypothesis = unique_rules
