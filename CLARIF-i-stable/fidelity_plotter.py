# fidelity_plotter.py

import os
import re
import itertools as it

from argparse import ArgumentParser
from copy import deepcopy
from matplotlib import pyplot as plt
from statistics import mean, stdev

from utils import (
    compute_fidelity,
    find_bubble_swap_action,
    find_bubble_partial_swap_action,
    find_quick_swap_action,
    find_quick_partial_swap_action,
)
from api.Coach import Coach, ProactiveCoach, ReflexiveCoach
from api.State import State

COACH_ACTIONS = {
    "bf": find_bubble_swap_action,
    "bp": find_bubble_partial_swap_action,
    "qf": find_quick_swap_action,
    "qp": find_quick_partial_swap_action,
}

COACH_TYPES = {
    "a": Coach,
    "r": ReflexiveCoach,
    "p": ProactiveCoach,
}

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MARKERS = ("^", ">", "v", "<")

CONFIGS = filter(
    lambda c: c[0] or not c[1],i
    it.product(
        (True, False), (True, False), COACH_ACTIONS.keys(), list(COACH_TYPES.keys())[1:] # HACK: Make results uniform
    ),
)

CONFIG_COLORS = dict(zip(COACH_ACTIONS.keys(), zip(COLORS, MARKERS)))

STYLE_CONFIGS_MAP = lambda c: (
    *CONFIG_COLORS[c[2]],
    "dotted" if c[1] else "dashed" if c[0] else "solid",
)


def parse_trace_line(line: str) -> list[State]:
    return [State.from_str(s) for s in line.split("; ")[1:]]


def split_header(hdr_str: str) -> tuple[int, int]:
    return [int(n) for n in hdr_str.split("; ")]


def parse_traces(path: str) -> dict[int, list[list[State]]]:
    trace_list = []
    traces = dict()
    HEADER_PATTERN = re.compile(r"^\d+; \d+$")
    previous_line = None
    first_n = -1
    with open(path, "r") as file:
        for line in iter(file.readline, ""):
            if HEADER_PATTERN.match(line):
                n, i = split_header(line)
                if n == first_n and i == 0:
                    trace_list.append(deepcopy(traces))
                    traces = dict()  # Just in case...
                if first_n == -1:
                    first_n = n
                if i == 0:
                    traces[n] = []
                if previous_line != None:
                    traces[n].append(parse_trace_line(previous_line))
            previous_line = line
        trace_list.append(traces)  # Append last config
    return trace_list


def compute_fidelities(traces: list, coach_type: str = "r") -> list:
    """traces as parsed by `parse_traces()`"""
    fidelities = dict()
    for ts, config in zip(traces, CONFIGS):
        if (not config[0] and config[1]) or (coach_type != "a" and config[3] != coach_type):
            continue
        coach_action_fn = COACH_ACTIONS[config[2]]
        fidelities[config] = {
            n: [compute_fidelity(t, coach_action_fn) for t in trs]
            for n, trs in ts.items()
        }
    return fidelities


def plot_fidelities(fidelities: dict, save_path) -> None:
    config_to_str = (
        lambda c: f"{c[2]}, Mem: {'Long' if c[1] else 'Short' if c[0] else 'None'}"
    )
    fig, ax = plt.subplots()
    ax.set_ylim((-0.05, 1.05))
    handles = []
    for config, fidelity in fidelities.items():
        ns, mean_fs, std_fs = zip(
            *((n, mean(fs), stdev(fs)) for n, fs in fidelity.items())
        )
        mean_minus, mean_plus = zip(*((m - s, m + s) for m, s in zip(mean_fs, std_fs)))
        print(config)
        c, m, ls = STYLE_CONFIGS_MAP(config)
        ax.fill_between(ns, mean_minus, mean_plus, alpha=0.1, color=c, linewidth=0.0)
        (h,) = ax.plot(
            ns,
            mean_fs,
            label=config_to_str(config),
            color=c,
            linestyle=ls,
            marker=m,
            alpha=0.9,
        )
        handles.append(h)
    ax.grid()
    ax.legend(
        handles=sorted(handles, key=lambda h: h._label),
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(save_path)


def prepare_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-N",
        type=int,
        default=20,
        help="Max state size. Default value: 20",
    )
    parser.add_argument(
        "-r",
        type=int,
        default=20,
        help="Reps count (per value of n). Default value: 20",
    )
    parser.add_argument(
        "-s",
        type=int,
        default=5,
        help="State size step. Default value: 5",
    )
    parser.add_argument(
        "-ct",
        type=str,
        default="r",
        choices=["a", "r", "p"],
        help="Coach type: re{a}ctive, {r}eflexive, {p}roactive. Default value: 'r'",
    )
    return parser


def main():
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(CWD, "raw_results")
    PLOTS_PATH = os.path.join(CWD, "plots")
    parser = prepare_parser()
    N, r, s, ct = vars(parser.parse_args()).values()
    fname = f"fidelity_test_N{N}_reps{r}_step{s}"
    source_name = fname + ".trace"
    res_path = os.path.join(RESULTS_PATH, source_name)
    fig_name = fname + f"_coach_{ct}.pdf"
    fig_path = os.path.join(PLOTS_PATH, fig_name)
    print("Parsing results...")
    traces = parse_traces(res_path)
    print("Magic happens...")
    fidelities = compute_fidelities(traces, coach_type=ct)
    print("Creating plot...")
    plot_fidelities(fidelities, fig_path)
    print(f"Plot saved at: {fig_path}")


if __name__ == "__main__":
    main()
