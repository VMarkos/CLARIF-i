import random
import itertools as it

from .api.TestCase import TestCase
from .api.State import State

def generate_sorting_test_case(n: int) -> TestCase:
    keys_s = ( f"k{i}" for i in range(n) )
    keys_g = it.tee(keys_s)
    start_state = State(dict(zip(keys_s, random.shuffle(list(range(n))))))
    goal_state = State(dict(zip(keys_g, range(n))))
    