# main.py
import os

from utils import generate_bubble_sort_test_case, generate_quick_sort_test_case
from api.Learner import Learner

ALGORITHMS = {
    'b': generate_bubble_sort_test_case,
    'q': generate_quick_sort_test_case,
}

def main():
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(CWD, "raw_results")
    algorithm = input("Enter algorithm ({q}uicksort, {b}ubblesort): ")
    N = int(input("Enter N: "))
    reps = int(input("Enter # of repetitions: "))
    memory = input("Remember advice (y/n): ")
    long_memory = "n"
    if memory == "y":
        long_memory = input("Remember across values of 'n' (y/n): ")
    res_file_name = os.path.join(RESULTS_PATH, f"{algorithm}_test_N{N}_reps{reps}_mem{memory}_long{long_memory}.txt")
    results_file = open(res_file_name, "w")
    results_file.write("")
    results_file.close()
    learner: Learner | None = Learner() if long_memory == "y" else None
    for n in range(1, N + 1):
        if long_memory == "n":
            learner = Learner() if memory == "y" else None
        for i in range(reps):
            print(f"Running test n={n}, rep={i}", end="\r")
            test = ALGORITHMS[algorithm](n, learner)
            test.run()
            results_file = open(res_file_name, "a")
            results_file.write(f"{n}; {test}\n")
            results_file.close()

if __name__ == "__main__":
    main()
