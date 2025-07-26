# api/TestCase

from copy import deepcopy

from .State import State
from .Learner import Learner
from .Coach import Coach, ReflexiveCoach
from .Rule import Rule
from typing import Callable

class TestCase:
    def __init__(self, start_state: State, is_goal: Callable[State, bool], target_rules: Callable[[State], Rule], learner: Learner | None=None, coach_class: Coach=Coach, full_reporting: bool = True, report_traces: bool = False) -> None:
        self.start_state: State = start_state
        # with open("log.txt", "a") as file:
        #     print(f"{self.start_state}", file=file)
        self.is_goal: State = is_goal
        self.learner: Learner = learner if learner != None else Learner()
        self.coach: Coach = coach_class(target_rules, is_goal, start_state)
        self.full_reporting: bool = full_reporting
        self._steps: int = 0
        self.report_traces: bool = self.full_reporting or report_traces
        self._learner_traces: list[list[State]] = []

    def run(self) -> None:
        # print(f"Start state: {self.start_state}")
        path = self.learner.search_path(self.start_state, self.is_goal)
        if self.report_traces:
            self._learner_traces.append(self.learner._trace)
        previous_advice = None
        while (advice := self.coach.evaluate_inference(path[1])) != ( True, [] ):
            # print(f"\t{[ str(s) for s in path[1] ]}")
            # print(f"\tArdvice {advice}")
            if previous_advice != None and all((x == y for x, y in zip(previous_advice, advice[1]))):
                print(f"\tLearner hypothesis: {self.learner.hypothesis}")
                raise ValueError(f"Duplicate advice:\n\t{advice}")
            self.learner.update_hypothesis(advice[1])
            path = self.learner.search_path(self.start_state, self.is_goal)
            if self.report_traces:
                self._learner_traces.append(self.learner._trace)
            previous_advice = deepcopy(advice[1])
            self._steps += 1

    def report(self) -> dict:
        return {
            "start_state": str(self.start_state) if self.full_reporting else "s",
            "learned_hypothesis": "; ".join(map(str, self.learner.hypothesis)) if self.full_reporting else "p",
            "steps": self._steps,
        }

    def get_traces_str(self) -> str:
        return "\n".join(("; ".join(str(s) for s in t) for t in self._learner_traces))

    def __str__(self) -> str:
        if self.full_reporting:
            attrs = [self._steps, self.start_state, self.learner.hypothesis]
        else:
            attrs = [self._steps, "s", "p"]
        return "; ".join(map(str, attrs))

