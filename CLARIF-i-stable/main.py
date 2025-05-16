# main.py

import itertools as it
from copy import deepcopy

from api.State import State
from api.Rule import Rule
from api.Learner import Learner
from api.Coach import Coach
from api.Action import Action

def find_swap_action(state: State, keys=['a', 'b', 'c']) -> Action | None:
    for i in range(len(keys) - 1):
        cur_key, next_key = keys[i], keys[i + 1]
        if state.get(cur_key) > state.get(next_key):
            def swap_callback(state: State):
                swapped_state = deepcopy(state)
                swapped_state.swap(cur_key, next_key)
                return swapped_state
            swap_action = Action(swap_callback)
            return swap_action
    return None

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
    states = ( State(dict(zip(KEYS, p))) for p in it.permutations(map(str, range(1, 4))) )
    target_rules = [
        Rule(
            f"R{i}",
            state,
            swap_action,
            priority=1,
            explanation=f"flipping", # maybe something more explicit
        )
        for i, state in enumerate(states)
        if (swap_action := find_swap_action(state)) != None
    ]
    # print("\n".join(map(str, target_rules)))
    coach = Coach(target_rules)
    # return
    steps = 0
    path = learner.search_path(start_state, goal_state)
    while (advice := coach.evaluate_inference(start_state, goal_state, path[1])) != ( True, []):
        # print(f">>> Advice[0] == {advice[1]}")
        # print(f"Hypothesis: {learner.hypothesis}", f"Advice: {advice}", sep="\n")
        learner.update_hypothesis(advice[1])
        path = learner.search_path(start_state, goal_state)
        # print("path:", [ (str(p[0]), [ (str(r[0]), r[1]) for r in p[1]]) for p in path[1]] )
        steps += 1
        # if steps == 4:
        #     print(">>> Returning due to step limit")
        #     return
    print(learner.hypothesis, steps, sep="\n")

if __name__ == "__main__":
    main()