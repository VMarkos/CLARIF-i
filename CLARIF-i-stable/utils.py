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

def find_quick_swap_action(state: State, keys: list[str]) -> Action:
    # print(f"State: {state}")
    n = len(keys)
    def quicksort(state: State, low: int = 0, high: int = n - 1) -> None:
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
    return quicksort(state) or Action()

def find_bubble_swap_action(state: State, keys: list[str]) -> Action:
    for i in range(len(keys) - 1):
        cur_key, next_key = keys[i], keys[i + 1]
        if state.get(cur_key) > state.get(next_key):
            def swap_callback(state: State):
                swapped_state = deepcopy(state)
                swapped_state.swap(cur_key, next_key)
                return swapped_state
            swap_action = Action(swap_callback, f"swap({cur_key}, {next_key})")
            return swap_action
    return Action()

def generate_bubble_sort_test_case(n: int, learner: Learner | None=None):
    return generate_sorting_test_case(n, find_bubble_swap_action, learner)

def generate_quick_sort_test_case(n: int, learner: Learner | None=None):
    return generate_sorting_test_case(n, find_quick_swap_action, learner)

def generate_sorting_test_case(n: int, action_fn: Callable[[State, list[str]], State], learner: Learner | None=None) -> TestCase:
    # Generate start and goal states
    digit_count = lambda n: 1 if 0 else int(math.log10(abs(n))) + 1
    pad_num = lambda n, p: '0' * (p - len((s := str(n)))) + s
    d = digit_count(n)
    keys = [ f"k{pad_num(i, d)}" for i in range(n) ]
    start_values = [ x for x in range(n) ]
    random.shuffle(start_values)
    start_state = State(dict(zip(keys, start_values)))
    goal_state = State(dict(zip(keys, [ x for x in range(n) ])))
    # print(f"Start: {start_state}\nGoal: {goal_state}")
    # Generate rules
    # states = ( State(dict(zip(keys, p))) for p in it.permutations(map(str, range(n))) )
    # TODO Rules need not be generated all at once, just a generator, or a something like that, since we have factorially many rules
    def get_triggered_rule(state: State) -> Rule:
        swap_action = action_fn(state, keys)
        # print(swap_action)
        return Rule(
            f"R({swap_action.name})",
            state,
            swap_action,
            priority=0,
            explanation=swap_action.name, # maybe something more explicit
        )
    # print("\n".join(map(str, target_rules)))
    test_case: TestCase = TestCase(start_state, goal_state, get_triggered_rule, learner)
    return test_case
        
