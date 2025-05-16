import random

from .api.TestCase import TestCase
from .api.State import State

def generate_sorting_test_case(n: int) -> TestCase:
    keys = ( f"k{i}" for i in range(n) )
    start_state = State(dict(zip(keys, random.shuffle())))
    goal_state = State(dict(zip(keys, range(n))))