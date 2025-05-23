# api/TestCase

from .State import State
from .Learner import Learner
from .Coach import Coach
from .Rule import Rule
from typing import Callable

class TestCase:
    def __init__(self, start_state: State, goal_state: State, target_rules: Callable[[State], Rule], learner: Learner | None=None) -> None:
        self.start_state: State = start_state
        self.goal_state: State = goal_state
        self.learner: Learner = learner if learner != None else Learner()
        self.coach: Coach = Coach(target_rules)
        self._steps: int = 0

    def run(self) -> None:
        path = self.learner.search_path(self.start_state, self.goal_state)
        while (advice := self.coach.evaluate_inference(self.start_state, self.goal_state, path[1])) != ( True, []):
            # print(f">>> Advice[0] == {advice[1]}")
            # print(f"Hypothesis: {self.learner.hypothesis}", f"Advice: {advice}", sep="\n")
            self.learner.update_hypothesis(advice[1])
            path = self.learner.search_path(self.start_state, self.goal_state)
            # print("path:", [ (str(p[0]), [ (str(r[0]), r[1]) for r in p[1]]) for p in path[1]] )
            self._steps += 1
            # if steps == 4:
            #     print(">>> Returning due to step limit")
            #     return
        # print(learner.hypothesis, steps, sep="\n")

    def report(self) -> dict:
        return {
            "start_state": str(self.start_state),
            "goal_state": str(self.goal_state),
            "learned_hypothesis": "; ".join(map(str, self.learner.hypothesis)),
            "steps": self._steps,
        }

    def __str__(self) -> str:
        attrs = [self._steps, self.start_state, self.goal_state, self.learner.hypothesis]
        return "; ".join(map(str, attrs))
