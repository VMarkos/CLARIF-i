# debug.py

import sys

from utils import generate_quick_sort_partial_test_case

from api.Learner import Learner
from api.State import State

from typing import Callable


def run_specific_test_case(
        n: int,
        learner: Learner | None,
        full_reporting: bool,
        start_state: State,
        goal_state: State,
        fn: Callable
    ) -> None:
    test = fn(n, learner, full_reporting, start_state, goal_state)
    print("Running test...")
    test.run()
    print("Test finished running!")
    report = test.report()
    print("=" * 30)
    print("REPORT")
    print(report["learned_hypothesis"].replace(";", "\n"))
    print("=" * 30)

def run_multiple_test_cases(
        n: int,
        full_reporting: bool,
        start_states: State,
        goal_states: State,
        fn: Callable,
        with_mem: bool,
    ) -> None:
    learner = None
    if with_mem:
        learner = Learner()
    for start_state, goal_state in zip(start_states, goal_states):
        print(f"start state: {start_state}")
        run_specific_test_case(n, learner, full_reporting, start_state, goal_state, fn)

def main():
    start_states: list[State] = []
    if len(sys.argv) > 2:
        print("E: Usage: `python3 debug.py` or `python3 debug.py <states_file>`", file=sys.stderr)
    elif len(sys.argv) == 2:
        states_filename = sys.argv[1]
        with open(states_filename, "r") as states_file:
            for line in states_file:
                state = State.from_str(line)
                start_states.append(state)
    else:
        start_state_str = input("Start state: ")
        state = State.from_str(start_state_str)
        start_states.append(state)
    n = len(start_states[0])
    with_mem = input("With memory (y/n): ") == "y"
    offset = int(input("Offset: "))
    goal_state = State(dict(zip(start_states[0].state.keys(), [ x for x in range(n) ])))
    goal_states = [goal_state] * len(start_states)
    run_multiple_test_cases(n, True, start_states[offset:], goal_states[offset:], generate_quick_sort_partial_test_case, with_mem)

if __name__ == "__main__":
    main()
