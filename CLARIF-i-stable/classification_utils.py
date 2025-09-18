# classification_utils.py

import numpy as np
import functools as ft
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
        return np.sqrt(self.dist_sq(other))

    @classmethod
    def from_str(cls, point_str) -> "Point":
        x, y = map(float, point_str.strip()[1:-1].split(","))
        return cls(x, y)

    def __str__(self) -> str:
        return f"({self.x},{self.y})"

    def __iter__(self) -> iter:
        return iter((self.x, self.y))

class Partition:
    def __init__(self, points: set[Point]=set(), k: int=5, tol: float=0.0) -> None:
        if points == set():
            self.points = set()
            self.k = 0
            self.parts = []
            self._tol = tol
            return
        self.points = deepcopy(points)
        self.k = k
        self._tol = tol # Used to quantify equality
        if k > len(self.points):
            raise ValueError(f"More partition classes than points: '{k} > {len(self.points)}'.")
        self.parts = self._initialise_parts()

    def _initialise_parts(self) -> set[set[Point]]:
        points_list = list(self.points)
        RNG.shuffle(points_list)
        return [ set(points_list[i::self.k]) for i in range(self.k) ]

    def update(self, other: "Partition") -> "Partition":
        # HACK: Is this really needed?
        self.points = other.points
        self.k = other.k
        self.parts = other.parts

    @classmethod
    def from_str(cls, partition_str: str) -> "Partition":
        part_strs = [ p[:p.rfind(',', -5)].strip().split(", ") for p in partition_str.split(":")[1:] ] # HACK: Find a more robust solution using regex
        parts = [ { Point.from_str(p) for p in ps } for ps in part_strs ]
        partition = cls.__new__(cls)
        partition.parts = parts
        partition.k = len(parts)
        partition.points = ft.reduce(lambda x, y: x.union(y), parts)
        return partition

    def get_part(self, p: Point) -> set[Point]:
        for i, part in enumerate(self.parts):
            if p in part:
                return i
        return -1

    def move(self, p: Point, from_part: int, to_part: int) -> None:
        if p not in self.parts[from_part]:
            p = self._find_closest_neighbour(p, self.parts[from_part])
        self.parts[from_part].remove(p)
        self.parts[to_part].add(p)

    def _find_closest_neighbour(self, p: Point, s: set) -> Point:
        for point in s:
            if p.dist(point) <= self._tol:
                return point
        raise ValueError(f"Point '{p}' has no neighbour within range 'self._tol'.")

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
        # TODO: this should be generalised to allow for comparisons in case of self._tol>0
        """ Two partitions are considered equal if their corresponding parts are 
        up to a factor of self._tol. """
        if not isinstance(other, Partition):
            return False
        return self.__key() == other.__key()

    def __hash__(self) -> int:
        return hash(self.__key())

    def __le__(self, other: "Partition") -> bool:
        if not isinstance(other, Partition):
            return False
        # other_parts = set(other.parts)
        return all(d <= self._tol for d in (min(hausdorff_distance(sp, op) for op in other.parts) for sp in self.parts))
        # return all(p in other.parts for p in self.parts)


    def __str__(self) -> str:
        parts_str = ', '.join(str(i) + ": " + ', '.join(map(str, part)) for i, part in enumerate(self.parts))
        return f"( {parts_str} )"

def hausdorff_distance(xs, ys) -> float:
    d_x = max(min(y.dist(x) for y in ys) for x in xs)
    d_y = max(min(y.dist(x) for x in xs) for y in ys)
    return max(d_x, d_y)

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

def generate_inertia_test_case(n: int, N: int=-1, learner: Learner | None=None, coach_class: Coach=Coach, full_reporting: bool=True, report_traces: bool=True, randomize_target: bool=False):
    return generate_classification_test_case(n, find_classification_inertia_action, N, learner, coach_class, full_reporting, report_traces)

def generate_classification_test_case(n: int, action_fn: Callable, N: int=20, learner: Learner | None=None, coach_class: Coach=Coach, full_reporting: bool=True, report_traces: bool=True, keep_advice_track: bool=False) -> TestCase:
    points = get_points(n)
    start_partition = Partition(points, k=4, tol=0.5e-1)
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
        keep_advice_track,
    )
    return test_case
