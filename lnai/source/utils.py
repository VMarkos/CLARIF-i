# utils.py
#
# General utilities

import numpy as np
import inspect
from functools import wraps
from copy import deepcopy
from lnai.rl.CurriculumSortEnv import CurriculumSortEnv


class SequenceMonitor:
    def __init__(self, obj, log_dict, arg_name) -> None:
        object.__setattr__(self, '_obj', obj)
        object.__setattr__(self, '_log_dict', log_dict)
        object.__setattr__(self, '_arg_name', arg_name)
        log_dict[arg_name] = [ deepcopy(obj) ]


    def __getattribute__(self, name):
        obj = object.__getattribute__(self, '_obj')
        log_dict = object.__getattribute__(self, '_log_dict')
        arg_name = object.__getattribute__(self, '_arg_name')
        attr = getattr(obj, name)
        if not callable(attr):
            return attr
        MUTATORS = {
            '__setitem__', '__delitem__', '__iadd__', '__imul__',
            'append', 'extend', 'insert', 'pop', 'remove', 'clear', 'sort'
        }
        if name in MUTATORS:
            @wraps(attr)
            def wrapper(*args, **kwargs):
                retval = attr(*args, **kwargs)
                log_dict[arg_name].append( deepcopy(obj) )
                return retval
            return wrapper
        return attr


    def __getitem__(self, item):
        return object.__getattribute__(self, '_obj')[item]


    def __setitem__(self, key, value):
        obj = object.__getattribute__(self, '_obj')
        log_dict = object.__getattribute__(self, '_log_dict')
        arg_name = object.__getattribute__(self, '_arg_name')
        obj[key] = value
        log_dict[arg_name].append( deepcopy(obj) )


    def __len__(self):
        return len(object.__getattribute__(self, '_obj'))


    def __repr__(self):
        obj = object.__getattribute__(self, '_obj')
        return f'SequenceMonitor({repr(obj)})'


def monitor_sequences(log_dict):
    def decorator(func):
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for k, v in bound_args.arguments.items():
                if hasattr(v, '__getitem__'):
                    bound_args.arguments[k] = SequenceMonitor(v, log_dict, k)
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator


# Sorting algorithms

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        # The last i elements are already in place
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # If no swaps occurred during this pass, the list is already sorted
        if not swapped:
            break
    return arr


def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
        
    if low < high:
        pi = _partition(arr, low, high)
        quick_sort(arr, low, pi - 1)  # Recursively sort left of pivot
        quick_sort(arr, pi + 1, high) # Recursively sort right of pivot
    return arr

def _partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            
    # Place the pivot in its correct sorted position
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def test_agent(model, stage: int = 5, max_n: int = 10):
    print(f"\n--- Testing Agent on Stage N = {stage} ---")
    env = CurriculumSortEnv(max_n=max_n)
    env.set_max_stage(stage)

    obs, _ = env.reset()
    env.current_n = stage
    # Fixed active permutation for a reproducible smoke test.
    env.state = np.arange(max_n, dtype=np.int64)
    env.state[:stage] = np.array([4, 1, 3, 0, 2, 5, 6, 7, 8, 9][:stage], dtype=np.int64)
    env._prev_tau = env._get_tau()
    obs = env.state.copy()

    print(f"Initial Vector: {env.state[:stage]}")

    done = False
    step_count = 0
    info = dict()
    while not done and step_count < 20:
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        i, j = env.action_to_swap(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        print(f"Step {step_count}: Swapped ({i}, {j}) -> State: {env.state[:stage]}")

    if info.get("is_success", False):
        print("[TEST RESULT]: SUCCESS! Vector successfully sorted.")
    else:
        print("[TEST RESULT]: FAILED.")
