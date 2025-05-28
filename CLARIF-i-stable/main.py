# main.py
import os
import math

from utils import generate_bubble_sort_test_case, generate_quick_sort_test_case, generate_bubble_sort_partial_test_case, generate_quick_sort_partial_test_case
from api.Learner import Learner

ALGORITHMS = {
    'b': generate_bubble_sort_test_case,
    'q': generate_quick_sort_test_case,
    'bp': generate_bubble_sort_partial_test_case,
    'qp': generate_quick_sort_partial_test_case,
}

def main():
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(CWD, "raw_results")
    algorithm = input("Enter algorithm ({q}uicksort, {b}ubblesort, append {p}artial): ")
    N = int(input("Enter N: "))
    reps = int(input("Enter # of repetitions: "))
    memory = input("Remember advice (y/n): ")
    long_memory = "n"
    if memory == "y":
        long_memory = input("Remember across values of 'n' (y/n): ")
    full_reporting = input("Report full policies (y/n): ") == "y"
    report_traces = False
    if not full_reporting:
        report_traces = input("Report traces (y/n): ") == "y"
    res_file_name = os.path.join(RESULTS_PATH, f"{algorithm}_test_N{N}_reps{reps}_mem{memory}_long{long_memory}.txt")
    trace_file_name = os.path.join(RESULTS_PATH, f"{algorithm}_test_N{N}_reps{reps}_mem{memory}_long{long_memory}.trace")
    with open(res_file_name, "w") as results_file:
        results_file.write("")
    if report_traces:
        with open(trace_file_name, "w") as trace_file:
            trace_file.write("")
    learner: Learner | None = Learner() if long_memory == "y" else None
    digit_count = lambda n: 1 if n == 0 else int(math.log10(n)) + 1
    trailing_spaces = " " * digit_count(N)
    for n in range(1, N + 1):
        if long_memory == "n":
            learner = Learner() if memory == "y" else None
        for i in range(reps):
            print(f"Running test n={n}, rep={i}", end=f"{trailing_spaces}\r")
            test = ALGORITHMS[algorithm](n, learner, full_reporting, report_traces)
            test.run()
            with open(res_file_name, "a") as results_file:
                results_file.write(f"{n}; {test}\n")
            if report_traces:
                with open(trace_file_name, "a") as trace_file:
                    trace_file.write(f"{n}; {i};\n{learner.get_traces_str()}")

if __name__ == "__main__":
    main()
