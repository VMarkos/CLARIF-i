import random
import itertools as it
from copy import deepcopy
from typing import Callable

from api.TestCase import TestCase
from api.State import State
from api.Rule import Rule
from api.Action import Action

def find_bubble_swap_action(state: State, keys) -> Action | None:
    for i in range(len(keys) - 1):
        cur_key, next_key = keys[i], keys[i + 1]
        if state.get(cur_key) > state.get(next_key):
            def swap_callback(state: State):
                swapped_state = deepcopy(state)
                swapped_state.swap(cur_key, next_key)
                return swapped_state
            swap_action = Action(swap_callback, f"swap({cur_key}, {next_key})")
            return swap_action
    return None

def generate_bubble_sort_test_case(n: int):
    return generate_sorting_test_case(n, find_bubble_swap_action)

def generate_sorting_test_case(n: int, action_fn: Callable[[State, list[str]], State]) -> TestCase:
    # Generate start and goal states
    keys = [ f"k{i}" for i in range(n) ]
    start_values = [ str(x) for x in range(n) ]
    random.shuffle(start_values)
    start_state = State(dict(zip(keys, start_values)))
    goal_state = State(dict(zip(keys, [ str(x) for x in range(n) ])))
    # print(f"Start: {start_state}\nGoal: {goal_state}")
    # Generate rules
    states = ( State(dict(zip(keys, p))) for p in it.permutations(map(str, range(n))) )
    # TODO Rules need not be generated all at once, just a generator, or a something like that, since we have factorially many rules
    target_rules = [
        Rule(
            f"R{i}",
            state,
            swap_action,
            priority=1,
            explanation=f"flipping", # maybe something more explicit
        )
        for i, state in enumerate(states)
        if (swap_action := action_fn(state, keys)) != None
    ]
    # print("\n".join(map(str, target_rules)))
    test_case: TestCase = TestCase(start_state, goal_state, target_rules)
    return test_case
    