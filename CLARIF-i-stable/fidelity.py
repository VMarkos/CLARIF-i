# fidelity.py

import logging
import os
import itertools as it
from argparse import ArgumentParser
from loggers import logger
from tqdm import tqdm

from utils import (
    generate_bubble_sort_test_case,
    generate_quick_sort_test_case,
    generate_bubble_sort_partial_test_case,
    generate_quick_sort_partial_test_case,
    generate_bubble_sort_test_case_at_k,
    generate_quick_sort_test_case_at_k,
)

from api.TestCase import TestCase
from api.Coach import Coach, ReflexiveCoach, ProactiveCoach
from api.Learner import Learner


def prepare_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-N",
        type=int,
        default=20,
        help=f"Maximum state size, N, for which the tests should be run. Default value: 20.",
    )
    parser.add_argument(
        "-r",
        type=int,
        default=20,
        help=f"Number of test condition repetitions for each value of 'n'. Default value: 20.",
    )
    parser.add_argument(
        "-s",
        type=int,
        default=5,
        help=f"Step value for 'n'. Default value: 5",
    )
    return parser


def single_run(
    algorithm,
    N,
    n_range,
    reps,
    memory,
    long_memory,
    coach_class,
    res_file_name,
    trace_file_name,
) -> None:
    learner = Learner() if long_memory else None
    logger.info(
        "Starting new series of experiments with configuration:\n"
        f"\talgorithm: {algorithm.__name__}\n"
        f"\tcoach type: {coach_class.__name__}\n"
        f"\tN: {N}\n"
        f"\treps: {reps}\n"
        f"\tshort memory: {memory}\n"
        f"\tlong memory: {long_memory}\n"
    )
    for n in n_range:
        logger.info(f"Running test for n={n}.")
        if not long_memory:
            learner = Learner() if memory else None
        for i in range(reps):
            test = algorithm(n, N, learner, coach_class, False, True, False)
            try:
                test.run()
            except ValueError as e:
                logger.warning(">>> Error while executing 'TestCase.run()': %s", e)
            with open(res_file_name, "a") as results_file:
                results_file.write(f"{n}; {test}\n")
            with open(trace_file_name, "a") as trace_file:
                trace_file.write(f"{n}; {i}\n{test.get_traces_str()}\n")
        logger.info("===== Iteration over! ======")
    logger.info(f"{'=' * 10} TEST OVER {'=' * 10}")


def main():
    # Initialize global paths
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(CWD, "raw_results")
    LOGS_PATH = os.path.join(CWD, "logs")
    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)
    if not os.path.isdir(LOGS_PATH):
        os.mkdir(LOGS_PATH)
    # Initialize internal configurations
    ALGORITHMS = (
        generate_bubble_sort_test_case,
        generate_bubble_sort_partial_test_case,
        generate_quick_sort_test_case,
        generate_quick_sort_partial_test_case,
    )
    COACHES = (
        Coach,
        ReflexiveCoach,
        ProactiveCoach,
    )
    PARTIAL_ALGORITHMS = (
        generate_bubble_sort_test_case_at_k,
        generate_quick_sort_test_case_at_k,
    )
    configs = list(filter(
        lambda c: c[0] or not c[1],
        it.product((True, False), (True, False), PARTIAL_ALGORITHMS, COACHES),
    ))
    # Create argument parser
    parser = prepare_parser()
    args = vars(parser.parse_args())
    N, r, s = args.values()
    # Result and trace files
    fname = f"fidelity_test_N{N}_reps{r}_step{s}_coaches{len(COACHES)}_partial_2_{N}"
    res_file_name = os.path.join(RESULTS_PATH, f"{fname}.txt")
    trace_file_name = os.path.join(RESULTS_PATH, f"{fname}.trace")
    FORMAT = "%(asctime)s :: %(message)s"
    log_file_name = os.path.join(LOGS_PATH, f"{fname}.log")
    logging.basicConfig(filename=log_file_name, format=FORMAT, level=logging.INFO)
    # Range of values for n
    n_range = range(s, N + s, s)
    # Initialize results and traces files
    with open(res_file_name, "w") as results_file:
        results_file.write("")
    with open(trace_file_name, "w") as trace_file:
        trace_file.write("")
    for k in range(2, N + 1):
        print(f"Running case for k={k}...")
        for short, long, algorithm, coach_class in tqdm(configs, total=18):
            # print(
            #     f"Running configuration: short memory: {short}, long memory: {long}, algorithm: {algorithm.__name__}"
            # )
            single_run(
                algorithm,
                N,
                n_range,
                r,
                short,
                long,
                coach_class,
                res_file_name,
                trace_file_name,
            )


if __name__ == "__main__":
    main()
