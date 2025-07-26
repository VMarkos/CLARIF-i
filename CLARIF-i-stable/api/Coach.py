# api/Coach.py

from typing import Callable
import random

from .State import State
from .Rule import Rule

from copy import deepcopy
from ordered_set import OrderedSet

class Coach:
    """
    The coach maintains target rules and provides feedback to the learner.
    
    Attributes:
        target_rules: List of rules that represent the desired behavior
    """

    def __init__(self, target_rules: Callable[[State], Rule], is_goal: Callable[State, bool], start_state: State) -> None:
        """ Initialise the coach with target rules """
        # self.target_rules: dict[State, Rule] = { rule.condition: rule for rule in sorted(target_rules, reverse=True) }
        self.target_rules = target_rules
        self.is_goal = is_goal
        self.start_state = start_state
        self._is_open_ended = is_goal is not None
    
    def evaluate_inference(self, traces: list[list[tuple[State, Rule | None]]]) -> tuple[bool, list[Rule]]:
        # print("Traces:", len(traces))
        if not traces:
            # print('\tNo goal no traces')
            return False, self._generate_goal_rules()
        if any(self.is_goal(trace[0]) for trace in traces):
            # print('\tGOAL')
            return True, []
        
        # print("\ttraces[-1]", traces[-1][0], [ (str(s), str(r)) for s, r in traces[-1][1] ])
        # print('\tNO goal but traces')
        return False, self._generate_goal_rules(self._pick_deviation_state(traces))
        
    def _pick_deviation_state(self, traces) -> State:
        return traces[-1][0]

    def _generate_goal_rules(self, current_state: State | None=None) -> list[Rule]:
        if current_state == None:
            current_state = self.start_state
        feedback_rules: list[Rule] = []
        # print(f"Current state: {current_state}")
        # print("Target rules:","\n".join(map(str, self.target_rules.keys())))
        # print(current_state in self.target_rules)
        advised_rule = self.target_rules(current_state)
        # print(f"\t>>> Advised rule: {advised_rule}")
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

class ReflexiveCoach(Coach):
    def __init__(self, target_rules: Callable[[State], Rule], is_goal: Callable[State, bool], start_state: State) -> None:
        super().__init__(target_rules, is_goal, start_state)
        self._cache_advice(target_rules)

    # a reflexive agent receives feedback, picks a random deviating point in the learner's traces and
    # provides a piece of advice for that stage.
    # So, the coach should maintain a way of solving the task at hand in beforehand, to speed things
    # up a bit
    # So this can be precomputed during initialisation.
    # Which means that the start state should be known during initialisation, so maybe 
    # pass it as an argument from `api.TestCase.run()`
    # The only difference lies in how the next rule is chosen, i.e., in line 37:
    # return False, self._generate_goal_rules(traces[-1][0])
    #                                         ^^^^^^^^^^^^^
    # This should be changed for each coach accordingly, so each class inherits the base class `Coach` and picks this 
    # using a separate `pick_advice()` method.

    def _cache_advice(self, target_rules) -> None:
        state_actions = dict()
        current_state = self.start_state
        while not self.is_goal(current_state):
            current_rule = self.target_rules(current_state)
            state_actions[current_state] = current_rule
            current_state = current_rule.apply(current_state)
        self._action_cache = state_actions
        # self.target_rules = lambda s: self._action_cache[s] # since things are cached, we need not run the algorithm again
        self.target_rules = self._cache_target_rules(target_rules)

    def _cache_target_rules(self, target_rules) -> Rule:
        def cached_tr(state: State) -> Rule:
            try:
                return self._action_cache[state]
            except KeyError:
                self._action_cache[state] = target_rules(state)
                return self._action_cache[state]
        return cached_tr

    def _pick_deviation_state(self, traces) -> State:
        last_path = traces[-1][1]
        # print(f"last path: {last_path}")
        if len(last_path) == 0:
            return self.start_state
        # print(f"action cache: {self._action_cache}")
        deviating_choices = OrderedSet([])
        previous_state = self.start_state
        # print(f"last path: {last_path}")
        for state, rule in last_path:
            if previous_state not in self._action_cache.keys() or self._action_cache[previous_state] != rule:
                deviating_choices.add(previous_state)
            previous_state = state
        if deviating_choices:
            # print(f"Deviations: {deviating_choices}")
            return random.choice(deviating_choices)
        # If there are no deviating choices, just ask for more advice
        # print(f"Last wrong state: {traces[-1][0]}")
        return traces[-1][0]
