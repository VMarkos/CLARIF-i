# api/TestCase

from copy import deepcopy

from .State import State
from .Learner import Learner
from .Coach import Coach
from .Rule import Rule
from typing import Callable

class TestCase:
    def __init__(self, start_state: State, goal_state: State, target_rules: Callable[[State], Rule], learner: Learner | None=None, full_reporting: bool = True) -> None:
        self.start_state: State = start_state
        # with open("log.txt", "a") as file:
        #     print(f"{self.start_state}", file=file)
        self.goal_state: State = goal_state
        self.learner: Learner = learner if learner != None else Learner()
        self.coach: Coach = Coach(target_rules)
        self.full_reporting: bool = full_reporting
        self._steps: int = 0

    def run(self) -> None:
        path = self.learner.search_path(self.start_state, self.goal_state)
        # print(f"Initial path: {path}")
        previous_advice = None
        while (advice := self.coach.evaluate_inference(self.start_state, self.goal_state, path[1])) != ( True, [] ):
            # print(f">>> Advice[1] == {advice[1]}")
            # print(f">>> Advice[0] == {advice[0]}")
            # print(f"path[0]: {path[0]}")
            if previous_advice != None and all((x == y for x, y in zip(previous_advice, advice[1]))):
                raise ValueError(f"Duplicate advice:\n\t{advice}")
            # print(f"Hypothesis: {self.learner.hypothesis}", f"Advice: {advice}", sep="\n")
            self.learner.update_hypothesis(advice[1])
            path = self.learner.search_path(self.start_state, self.goal_state)
            # print("path:", [ (str(p[0]), [ (str(r[0]), r[1]) for r in p[1]]) for p in path[1]] )
            previous_advice = deepcopy(advice[1])
            self._steps += 1
            # if steps == 4:
            #     print(">>> Returning due to step limit")
            #     return
        # print(learner.hypothesis, steps, sep="\n")

    def report(self) -> dict:
        return {
            "start_state": str(self.start_state) if self.full_reporting else "s",
            "goal_state": str(self.goal_state) if self.full_reporting else "g",
            "learned_hypothesis": "; ".join(map(str, self.learner.hypothesis)) if self.full_reporting else "p",
            "steps": self._steps,
        }

    def __str__(self) -> str:
        if self.full_reporting:
            attrs = [self._steps, self.start_state, self.goal_state, self.learner.hypothesis]
        else:
            attrs = [self._steps, "s", "g", "p"]
        return "; ".join(map(str, attrs))

