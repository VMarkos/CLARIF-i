# main.py
import os
import math
import logging
from tqdm import tqdm

from loggers import logger
from utils import (
    generate_bubble_sort_test_case,
    generate_quick_sort_test_case,
    generate_bubble_sort_partial_test_case,
    generate_quick_sort_partial_test_case,
)
from classification_utils import generate_inertia_test_case
from api.Learner import Learner
from api.Coach import Coach, ReflexiveCoach, ProactiveCoach

ALGORITHMS = {
    "b": generate_bubble_sort_test_case,
    "q": generate_quick_sort_test_case,
    "bp": generate_bubble_sort_partial_test_case,
    "qp": generate_quick_sort_partial_test_case,
    "i": generate_inertia_test_case,
}

COACHES = {
    "a": Coach,
    "f": ReflexiveCoach,
    "p": ProactiveCoach,
}


def main():
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(CWD, "raw_results")
    LOGS_PATH = os.path.join(CWD, "logs")
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
    if not os.path.isdir(LOGS_PATH):
        os.mkdir(LOGS_PATH)
    algorithm = input(
        "Enter algorithm ({q}uicksort, {b}ubblesort, append {p}artial, {i}nertia clustering): "
    )
    coach_type = input("Enter coach type (re{a}ctive, re{f}lexive, {p}roactive): ")
    coach_class = COACHES[coach_type]
    randomize_targets = input("Randomize targets (y/n): ") == "y"
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
    res_file_name = os.path.join(
        RESULTS_PATH,
        f"{algorithm}_{coach_type}_test_N{N}_reps{reps}_mem{memory}_long{long_memory}.txt",
    )
    trace_file_name = os.path.join(
        RESULTS_PATH,
        f"{algorithm}_{coach_type}_test_N{N}_reps{reps}_mem{memory}_long{long_memory}.trace",
    )
    FORMAT = "%(asctime)s :: %(message)s"
    log_file_name = os.path.join(
        LOGS_PATH,
        f"{algorithm}_{coach_type}_test_N{N}_reps{reps}_mem{memory}_long{long_memory}.log",
    )
    logging.basicConfig(filename=log_file_name, format=FORMAT, level=logging.INFO)
    with open(res_file_name, "w") as results_file:
        results_file.write("")
    if report_traces:
        with open(trace_file_name, "w") as trace_file:
            trace_file.write("")
    learner: Learner | None = Learner() if long_memory == "y" else None
    digit_count = lambda n: 1 if n == 0 else int(math.log10(n)) + 1
    trailing_spaces = " " * digit_count(N)
    logger.info(
        "Starting new series of experiments with configuration:\n"
        f"\talgorithm: {algorithm}\n"
        f"\tcoach type: {coach_type}\n"
        f"\tN: {N}\n"
        f"\treps: {reps}\n"
        f"\tshort memory: {memory}\n"
        f"\tlong memory: {long_memory}\n"
    )
    n_range = range(5, N + 1, 5) if algorithm == "i" else range(1, N + 1)
    for n in n_range:
        print(f"Running test for n={n}")
        logger.info(f"Running test for n={n}")
        if long_memory == "n":
            learner = Learner() if memory == "y" else None
        for i in tqdm(range(reps)):
            test = ALGORITHMS[algorithm](
                n,
                N,
                learner,
                coach_class,
                full_reporting,
                report_traces,
                randomize_targets,
            )
            try:
                test.run()
            except ValueError as e:
                logger.warning(">>> Error while executing 'TestCase.run()': %s", e)
            with open(res_file_name, "a") as results_file:
                results_file.write(f"{n}; {test}\n")
            if report_traces:
                with open(trace_file_name, "a") as trace_file:
                    trace_file.write(f"{n}; {i}\n{test.get_traces_str()}\n")
        logger.info("===== Iteration over! ======")
    logger.info(f"{'=' * 10} TEST OVER {'=' * 10}")


if __name__ == "__main__":
    main()
