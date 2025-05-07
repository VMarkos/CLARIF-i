# main.py

import itertools as it
from copy import deepcopy

from api.State import State
from api.Rule import Rule
from api.Learner import Learner
from api.Coach import Coach

def flip_state(state: State, keys=['a', 'b', 'c']):
    """ Flips the first incorrect pair in a state returning a new state"""
    # print(type(state))
    for i in range(len(keys) - 1):
        cur_key, next_key = keys[i], keys[i + 1]
        if state.get(cur_key) > state.get(next_key):
            flipped_state = deepcopy(state)
            flipped_state.swap(cur_key, next_key)
            return flipped_state
    return deepcopy(state)

def main():
    # Initialise start and goal states
    start_state = State({
        "a": "2",
        "b": "3",
        "c": "1",
    })
    goal_state = State({
        "a": "1",
        "b": "2",
        "c": "3",
    })

    # Initialise learner
    learner = Learner()

    # Initialise target rules and coach
    KEYS = ['a', 'b', 'c']
    states = ( State(dict(zip(KEYS, p))) for p in it.permutations(range(1, 4)) )
    target_rules = [
        Rule(
            f"",
            state,
            flipped_state,
            priority=1,
            explanation=f"flipping", # maybe something more explicit
        )
        for state in states
        if (flipped_state := flip_state(state)) != state
    ]
    # print("\n".join(map(str, target_rules)))
    coach = Coach(target_rules)

    steps = 0
    path = learner.search_path(start_state, goal_state)
    while (advice := coach.evaluate_inference(start_state, goal_state, path[1])) != (True, []):
        learner.update_hypothesis(advice[1])
        steps += 1
    print(learner.hypothesis, steps, sep="\n")

if __name__ == "__main__":
    main()