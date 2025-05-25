# utils.py

import random
import math
import itertools as it
from copy import deepcopy
from typing import Callable

from api.TestCase import TestCase
from api.Learner import Learner
from api.Rule import Rule
from api.Action import Action
from api.State import State

# To speed things up in all cases we need some sort of memory, e.g., remember some parameters for each algorithm to save up time in rule generation
# These should not be kept into the state itself but maybe some of the agents (learner? coach? TestCase? `target_rules` itself?)

def find_quick_swap_action(state: State, keys: list[str]) -> tuple[State, Action, int]:
    # print(f"State: {state}")
    n = len(keys)
    def quicksort(state: State, low: int = 0, high: int = n - 1) -> int | Action:
        if low >= 0 and high >= 0 and low < high:
            p = partition(state, low, high)
            if isinstance(p, Action):
                return p
            low_action = quicksort(state, low, p)
            if low_action != None:
                return low_action
            high_action = quicksort(state, p + 1, high)
            if high_action != None:
                return high_action
    # Define partition
    def partition(state: State, low: int, high: int) -> int | Action:
        pivot = state.get(keys[low])
        i = low
        j = high
        while True:
            while i <= high and state.get(keys[i]) < pivot:
                i += 1
            while j >= low and state.get(keys[j]) > pivot:
                j -= 1
            if i >= j:
                return j
            left_key = keys[i]
            right_key = keys[j]
            def swap_callback(state: State):
                swapped_state = deepcopy(state)
                swapped_state.swap(left_key, right_key)
                return swapped_state
            swap_action = Action(swap_callback, f"swap({left_key}, {right_key})")
            # print("\tswap action", swap_action)
            return swap_action
    action = quicksort(state) or Action()
    return state, action, 0

def find_quick_partial_swap_action(state: State, keys: list[str]) -> tuple[State, Action, int]:
    # print(f"State: {state}")
    n = len(keys)
    priority: int = n * n
    dec_priority = lambda t: t[:-1] + (t[-1] - 1, )
    def quicksort(state: State, low: int = 0, high: int = n - 1, priority: int = priority) -> int | Action:
        if low >= 0 and high >= 0 and low < high:
            p = partition(state, low, high, priority)
            priority -= 1
            if isinstance(p, tuple):
                action = dec_priority(p)
                return p
            low_action = quicksort(state, low, p, priority)
            if low_action != None:
                action = dec_priority(low_action)
                return action
            high_action = quicksort(state, p + 1, high, priority)
            if high_action != None:
                action = dec_priority(high_action)
                return high_action
        priority -= 1
    # Define partition
    def partition(state: State, low: int, high: int, priority: int) -> int | tuple[Action, int, int, int]:
        pivot = state.get(keys[low])
        i = low
        j = high
        while True:
            while i <= high and state.get(keys[i]) < pivot:
                i += 1
                priority -= 1
            while j >= low and state.get(keys[j]) > pivot:
                j -= 1
                priority -=1
            if i >= j:
                priority -= 1
                return j
            left_key = keys[i]
            right_key = keys[j]
            def swap_callback(state: State):
                swapped_state = deepcopy(state)
                swapped_state.swap(left_key, right_key)
                return swapped_state
            swap_action = Action(swap_callback, f"swap({left_key}, {right_key})")
            # print("\tswap action", swap_action)
            priority -= 1
            return swap_action, left_key, right_key, priority
    action, left_key, right_key, priority = quicksort(state) or (Action(), None, None, 0)
    swap_state = State({ left_key: state.get(left_key), right_key: state.get(right_key) }) if action != None else State()
    priority -= 1
    return swap_state, action, priority

def find_bubble_swap_action(state: State, keys: list[str]) -> tuple[State, Action, int]:
    for i in range(len(keys) - 1):
        cur_key, next_key = keys[i], keys[i + 1]
        if state.get(cur_key) > state.get(next_key):
            def swap_callback(state: State):
                swapped_state = deepcopy(state)
                swapped_state.swap(cur_key, next_key)
                return swapped_state
            swap_action = Action(swap_callback, f"swap({cur_key}, {next_key})")
            return state, swap_action, 0
    return state, Action(), 0

def find_bubble_partial_swap_action(state: State, keys: list[str]) -> tuple[State, Action, int]:
    n = len(keys)
    for i in range(n - 1):
        cur_key, next_key = keys[i], keys[i + 1]
        cur_val = state.get(cur_key)
        next_val = state.get(next_key)
        if cur_val > next_val:
            def swap_callback(state: State):
                swapped_state = deepcopy(state)
                swapped_state.swap(cur_key, next_key)
                return swapped_state
            swap_state = State({ cur_key: cur_val, next_key: next_val })
            swap_action = Action(swap_callback, f"swap({cur_key}, {next_key})")
            return swap_state, swap_action, n - i
    return State(), Action(), 0 # Maybe this should return the full state?
    
def generate_bubble_sort_partial_test_case(n: int, learner: Learner | None=None, full_reporting: bool = True, start_state: State = None, goal_state: State = None):
    return generate_sorting_test_case(n, find_bubble_partial_swap_action, learner, full_reporting, start_state, goal_state)

def generate_quick_sort_partial_test_case(n: int, learner: Learner | None=None, full_reporting: bool = True, start_state: State = None, goal_state: State = None):
    return generate_sorting_test_case(n, find_quick_partial_swap_action, learner, full_reporting, start_state, goal_state)

def generate_bubble_sort_test_case(n: int, learner: Learner | None=None, full_reporting: bool = True, start_state: State = None, goal_state: State = None):
    return generate_sorting_test_case(n, find_bubble_swap_action, learner, full_reporting, start_state, goal_state)

def generate_quick_sort_test_case(n: int, learner: Learner | None=None, full_reporting: bool = True, start_state: State = None, goal_state: State = None):
    return generate_sorting_test_case(n, find_quick_swap_action, learner, full_reporting, start_state, goal_state)

def generate_sorting_test_case(n: int, action_fn: Callable[[State, list[str]], State], learner: Learner | None=None, full_reporting: bool = True, start_state: State = None, goal_state: State = None) -> TestCase:
    # Generate start and goal states
    digit_count = lambda n: 1 if 0 else int(math.log10(abs(n))) + 1
    pad_num = lambda n, p: '0' * (p - len((s := str(n)))) + s
    d = digit_count(n)
    keys = [ f"k{pad_num(i, d)}" for i in range(n) ]
    start_values = [ x for x in range(n) ]
    random.shuffle(start_values)
    start_state = State(dict(zip(keys, start_values))) if start_state == None else start_state
    goal_state = State(dict(zip(keys, [ x for x in range(n) ]))) if goal_state == None else goal_state
    # print(f"Start: {start_state}\nGoal: {goal_state}")
    # Generate rules
    # states = ( State(dict(zip(keys, p))) for p in it.permutations(map(str, range(n))) )
    # TODO Rules need not be generated all at once, just a generator, or a something like that, since we have factorially many rules
    def get_triggered_rule(state: State) -> Rule:
        action_state, swap_action, priority = action_fn(state, keys)
        # print(swap_action)
        return Rule(
            f"R({swap_action.name})",
            action_state,
            swap_action,
            priority=priority,
            explanation=swap_action.name, # maybe something more explicit
        )
    # print("\n".join(map(str, target_rules)))
    test_case: TestCase = TestCase(start_state, goal_state, get_triggered_rule, learner, full_reporting)
    return test_case
        
