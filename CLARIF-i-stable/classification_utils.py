# classification_utils.py

import random
from copy import deepcopy
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

class Partition:
    def __init__(self, points: list[Point], k: int=5) -> None:
        self.points = deepcopy(points)
        self.k = k
        if k > len(self.points):
            raise ValueError(f"More partition classes than points: '{k} > {len(self.points)}'")
        self.parts = self._initialise_parts()

    def _initialise_parts(self) -> set[set[Point]]]:
        random.shuffle(self.points)
        return { set(self.points[i::self.k]) for i in range(self.k) }

    def _compute_inertia(self, p: Point, partition: set[Point]) -> float:
        pass

    def _find_best_fit(self, p: Point) -> set[Point]:
        pass

def find_classification_inertia_action(partition: Partition) -> tuple[Partition, Action, int]:
    pass

# TODO: What else?
