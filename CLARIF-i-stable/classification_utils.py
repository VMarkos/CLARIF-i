# classification_utils.py

import random
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

from api.TestCase import TestCase
from api.Learner import Learner
from api.Coach import Coach
from api.Rule import Rule
from api.Action import Action

SEED = 164595198122839924703705421391598440892
RNG = np.random.default_rng(SEED)

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def dist_sq(self, other: "Point") -> float:
        return (self.x - other.x) ** 2 + (self.y - other.y) ** 2

    def dist(self, other: "Point") -> float:
        return sqrt(self.dist_sq(other))

class Partition:
    def __init__(self, points: set[Point], k: int=5) -> None:
        self.points = deepcopy(points)
        self.k = k
        if k > len(self.points):
            raise ValueError(f"More partition classes than points: '{k} > {len(self.points)}'.")
        self.parts = self._initialise_parts()

    def _initialise_parts(self) -> set[set[Point]]:
        points_list = list(self.points)
        random.shuffle(points_list)
        return [ set(points_list[i::self.k]) for i in range(self.k) ]

    def get_part(self, p: Point) -> set[Point]:
        for i, part in enumerate(self.parts):
            if p in part:
                return i
        return -1

    def move(self, p: Point, from_part: int, to_part: int) -> None:
        self.parts[from_part].remove(p)
        self.parts[to_part].add(p)

    def _compute_inertia(self, p: Point, partition: set[Point]) -> float:
        return sum(p.dist_sq(x) for x in partition)

    def find_best_fit(self, p: Point) -> set[Point]:
        min_inertia, min_part = np.inf, None
        for i, part in enumerate(self.parts):
            cur_inertia = self._compute_inertia(p, part)
            if cur_inertia < min_inertia:
                min_inertia = cur_inertia
                min_part = i
        return min_part

    def __key(self) -> int:
        return tuple(tuple(p) for p in self.parts)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Partition):
            return False
        return self.__key() == other.__key()

    def __hash__(self) -> int:
        return hash(self.__key())

    def __str__(self) -> str:
        parts_str = ', '.join(str(i) + ": " + ', '.join(map(str, part)) for i, part in enumerate(self.parts))
        return f"( {parts_str} )"

def get_move_callback(partition: Partition, point: Point, from_part: int, to_part: int) -> Callable:
    def move_callback(partition: Partition):
        moved_partition = deepcopy(partition)
        moved_partition.move(point, from_part, to_part)
        return moved_partition
    return move_callback

def find_classification_inertia_action(partition: Partition) -> tuple[Partition, Action, int]:
    for p in partition.points:
        cur_part = partition.get_part(p)
        best_fit = partition.find_best_fit(p)
        if cur_part != best_fit:
            move_callback = get_move_callback(partition, p, cur_part, best_fit)
            move_action = Action(move_callback, f"move({p}, {cur_part}, {best_fit})")
            return partition, move_action, 0
    return partition, Action(), 0

def get_rule_selector(action_fn: Callable) -> Callable:
    def rule_selector(p: Partition) -> Rule:
        partition, move_action, priority = action_fn(p)
        return Rule(
            f"R({move_action.name})",
            partition,
            move_action,
            priority=priority,
            explanation=move_action.name,
        )
    return rule_selector

def get_points(n: int) -> set[Point]:
    """ Not uniformly random, due to resampling """
    points = set()
    while len(points) < n:
        p = Point(RNG.random(), RNG.random())
        if p not in points:
            points.add(p)
    return points

def generate_classification_test_case(n: int, action_fn: Callable, N: int=20, learner: Learner | None=None, coach_class: Coach=Coach, full_reporting: bool=True, report_traces: bool=True) -> TestCase:
    points = get_points(n)
    start_partition = Partition(points, k=4)
    is_goal = lambda p: all(p.get_part(x) == p.find_best_fit(x) for x in p.points)
    rule_selector = get_rule_selector(action_fn)
    test_case = TestCase(
        start_partition,
        is_goal,
        rule_selector,
        learner,
        coach_class,
        full_reporting,
        report_traces,
    )
    return test_case
