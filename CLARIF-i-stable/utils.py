# utils.py

import os
import random
import math
import json
import itertools as it
from copy import deepcopy
from typing import Callable

from api.TestCase import TestCase
from api.Learner import Learner
from api.Rule import Rule
from api.Action import Action
from api.State import State

# To speed things up in all cases we need some sort of memory, e.g., remember some parameters for each algorithm to save up time in rule generation
# Theseshould not be kept into the state itself but maybe some of the agents (learner? coach? TestCase? `target_rules` itself?)

# Local Lambdas
digit_count = lambda n: 1 if n == 0 else int(math.log10(abs(n))) + 1
pad_num = lambda n, p: '0' * (p - len((s := str(n)))) + s

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
            swap_callback = get_swap_callback(let_key, right_key)
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
            swap_callback = get_swap_callback(left_key, right_key)
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
            swap_callback = get_swap_callback(cur_key, next_key)
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
            swap_callback = get_swap_callback(cur_key, next_key)
            swap_state = State({ cur_key: cur_val, next_key: next_val })
            swap_action = Action(swap_callback, f"swap({cur_key}, {next_key})")
            return swap_state, swap_action, n - i
    return State(), Action(), 0

def _generate_targets(N: int=20) -> None:
    targets = dict()
    d = digit_count(N)
    for n in range(1, N + 1):
        targets[n] = { pad_num(k, d): v for k, v in enumerate(random.sample(range(1, n + 1), k=n)) }
    CWD = os.path.abspath(os.path.dirname(__file__))
    targets_path = os.path.join(CWD, 'targets.json')
    with open(targets_path, 'w') as file:
        json.dump(targets, file, indent=2)

def _load_targets() -> dict[int, dict[str, int]]:
    CWD = os.path.abspath(os.path.dirname(__file__))
    targets_path = os.path.join(CWD, 'targets.json')
    with open(targets_path, 'r') as file:
        targets = json.load(file)
    return targets

TARGETS = _load_targets()

def find_approximate_partial_swap_action(state: State, keys: list[str]) -> tuple[State, Action, int]:
    target = State(TARGETS[str(len(state))])
    state_keys = state.state.keys()
    threshold = 0.7
    current_kendall_tau = target.kendall_tau(state)
    if current_kendall_tau > threshold:
        return State(), Action(), int(1000 * current_kendall_tau)
    for l, r in it.product(state_keys, state_keys):
        if l >= r:
            continue
        copy_state = deepcopy(state)
        copy_state.swap(l, r)
        new_kendall_tau = target.kendall_tau(copy_state)
        if new_kendall_tau > current_kendall_tau:
            swap_callback = get_swap_callback(l, r)
            swap_state = State({ l: state.get(l), r: state.get(r) })
            swap_action = Action(swap_callback, f"swap({l}, {r})")
            return swap_state, swap_action, int(1000 * new_kendall_tau)
    return State(), Action(), 1000 # Typically, this should never be reached except for ill-defined settings
   
def get_swap_callback(left, right) -> Callable:
    def swap_callback(state: State):
        swapped_state = deepcopy(state)
        swapped_state.swap(left, right)
        return swapped_state
    return swap_callback

def generate_approximate_partial_test_case(n: int, N: int=20, learner: Learner | None=None, full_reporting: bool=True, report_traces: bool=True):
    return generate_sorting_test_case(n, find_approximate_partial_swap_action, N, learner, full_reporting, report_traces)

def generate_bubble_sort_partial_test_case(n: int, N: int=20, learner: Learner | None=None, full_reporting: bool = True, report_traces: bool = True):
    return generate_sorting_test_case(n, find_bubble_partial_swap_action, N, learner, full_reporting, report_traces)

def generate_quick_sort_partial_test_case(n: int, N: int=20, learner: Learner | None=None, full_reporting: bool = True, report_traces: bool = True):
    return generate_sorting_test_case(n, find_quick_partial_swap_action, N, learner, full_reporting, report_traces)

def generate_bubble_sort_test_case(n: int, N: int=20, learner: Learner | None=None, full_reporting: bool = True, report_traces: bool = True):
    return generate_sorting_test_case(n, find_bubble_swap_action, N, learner, full_reporting, report_traces)

def generate_quick_sort_test_case(n: int, N: int=20, learner: Learner | None=None, full_reporting: bool = True, report_traces: bool = True):
    return generate_sorting_test_case(n, find_quick_swap_action, N, learner, full_reporting, report_traces)

def generate_sorting_test_case(n: int, action_fn: Callable[[State, list[str]], State], N: int=20, learner: Learner | None=None, full_reporting: bool = True, report_traces: bool = True, is_approx: bool=False) -> TestCase:
    # Generate start and goal states
    d = digit_count(N)
    keys = [ f"k{pad_num(i, d)}" for i in range(n) ]
    start_values = [ x for x in range(n) ]
    random.shuffle(start_values)
    start_state = State(dict(zip(keys, start_values)))
    is_goal = None
    if is_approx:
        goal_state = State(TARGETS[str(n)])
        is_goal = lambda s: goal_state.kendall_tau(s) > 0.7
    else:
        goal_state = State(dict(zip(keys, [ x for x in range(n) ])))
        is_goal = lambda s: goal_state == s
    # print(f"Start: {start_state}\nGoal: {goal_state}")
    # Generate rules
    # states = ( State(dict(zip(keys, p))) for p in it.permutations(map(str, range(n))) )
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
    test_case: TestCase = TestCase(start_state, is_goal, get_triggered_rule, learner, full_reporting, report_traces)
    return test_case
        
