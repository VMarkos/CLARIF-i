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
    find_bubble_swap_action_at_k,
    find_quick_swap_action_at_k,
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
    lambda c: c[0] or not c[1],
    it.product(
        (True, False), (True, False), COACH_ACTIONS.keys(), list(COACH_TYPES.keys())[1:] # HACK: Make results uniform
    ),
)

PARTIAL_COACH_ACTIONS = {
    'b': find_bubble_swap_action_at_k,
    'q': find_quick_swap_action_at_k,
}

PARTIAL_CONFIGS = [
    c for c in it.product((True, False), (True, False), PARTIAL_COACH_ACTIONS.keys(), COACH_TYPES.keys())
    if c[0] or not c[1]
]

CONFIG_COLORS = dict(zip(COACH_ACTIONS.keys(), zip(COLORS, MARKERS)))

PARTIAL_CONFIG_COLORS = dict(zip(it.product(PARTIAL_COACH_ACTIONS.keys(), (2, 20)), zip(COLORS, MARKERS)))

STYLE_CONFIGS_MAP = lambda c: (
    *CONFIG_COLORS[c[2]],
    "dotted" if c[1] else "dashed" if c[0] else "solid",
)

PARTIAL_STYLE_CONFIGS_MAP = lambda c, k: (
    *PARTIAL_CONFIG_COLORS[(c[2], k)],
    "dotted" if c[1] else "dashed" if c[0] else "solid",
)

def parse_trace_line(line: str) -> list[State]:
    try:
        return [State.from_str(s) for s in line.split("; ")[1:]]
    except Exception as e:
        print("trace line:", line)
        raise e


def split_header(hdr_str: str) -> tuple[int, int]:
    return [int(n) for n in hdr_str.split("; ")]

def parse_traces(path: str) -> dict[int, list[list[State]]]:
    trace_list = []
    traces = []
    HEADER_PATTERN = re.compile(r"^\d+; \d+$")
    previous_line = None
    n = -1
    i = -1
    with open(path, "r") as file:
        for line in iter(file.readline, ""):
            if HEADER_PATTERN.match(line):
                n, i = split_header(line)
                if previous_line != None:
                    traces.append(parse_trace_line(previous_line))
                if i == 0:
                    # trace_list.append(deepcopy(traces))
                    if traces != []:
                        trace_list.append( (n, deepcopy(traces)) )
                    traces = []
            previous_line = line
        trace_list.append( (n, deepcopy(traces)) )  # Append last config
    return trace_list

K_STEPS = [4 * 18] * 4 + [3 * 18] * 5 + [2 * 18] * 5 + [1 * 18] * 5

def compute_partial_fidelities(traces: list, coach_type: str="a", k_range=range(2, 21)) -> list:
    fidelities = dict()
    # step = len(traces) // (k_range[-1] - k_range[0])
    # print(step, len(traces), traces[0])
    for k in k_range:
        offset = sum(K_STEPS[:(k - 2)])
        print(offset)
        fidelities[k] = compute_fidelities(traces, coach_type, configs=PARTIAL_CONFIGS, coach_actions=PARTIAL_COACH_ACTIONS, k=k)
    return fidelities

def compute_fidelities(traces: list, coach_type: str="r", configs=CONFIGS, coach_actions=COACH_ACTIONS, k: int=2) -> list:
    """traces as parsed by `parse_traces()`"""
    fidelities = dict()
    offset = sum(K_STEPS[:(k - 2)])
    step = K_STEPS[k - 2]
    # FIXME: Here you should group things by configuration by groupby or something like that...
    for ts, config in zip(traces[offset:(offset + step)], configs): # FIXME: This needs to be recalculated
        if (not config[0] and config[1]) or config[3] != coach_type:
            continue
        coach_action_fn = lambda s, ks: coach_actions[config[2]](s, ks, k=k)
        fidelities[config] = {
            n: [compute_fidelity(t, coach_action_fn) for t in trs]
            for n, trs in ts
        }
    return fidelities

def config_to_str(c: tuple) -> str:
    return f"{c[2]}, Mem: {'Long' if c[1] else 'Short' if c[0] else 'None'}"

def plot_partial_fidelities(fidelities: dict, save_path, k_range=range(2,21)) -> None:
    fig, ax = plt.subplots()
    ax.set_ylim((-0.05, 1.05))
    handles = []
    for k in k_range:
        for config, fidelity in fidelities[k].items():
            print(config)
            add_line(config, fidelity, ax, handles, k)
    ax.grid()
    ax.legend(
        handles=sorted(handles, key=lambda h: h._label),
        loc="upper right"
    )
    fig.tight_layout()
    fig.savefig(save_path)


def plot_fidelities(fidelities: dict, save_path) -> None:
    fig, ax = plt.subplots()
    ax.set_ylim((-0.05, 1.05))
    handles = []
    for config, fidelity in fidelities.items():
        add_line(config, fidelity, ax, handles)
    ax.grid()
    ax.legend(
        handles=sorted(handles, key=lambda h: h._label),
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(save_path)

def add_line(config, fidelity, ax, handles, k) -> None:
    ns, mean_fs, std_fs = zip(
        *((n, mean(fs), stdev(fs)) for n, fs in fidelity.items())
    )
    mean_minus, mean_plus = zip(*((m - s, m + s) for m, s in zip(mean_fs, std_fs)))
    # c, m, ls = STYLE_CONFIGS_MAP(config)
    c, m, ls = PARTIAL_STYLE_CONFIGS_MAP(config, k)
    ax.fill_between(ns, mean_minus, mean_plus, alpha=0.1, color=c, linewidth=0.0)
    (h,) = ax.plot(
        ns,
        mean_fs,
        label=config_to_str(config) + f"@{k}",
        color=c,
        linestyle=ls,
        marker=m,
        alpha=0.9,
    )
    handles.append(h)


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
        default="a",
        choices=["a", "r", "p"],
        help="Coach type: re{a}ctive, {r}eflexive, {p}roactive. Default value: 'r'",
    )
    parser.add_argument(
        "-ks",
        type=int,
        nargs='*',
        default=list(range(2, 21)),
        help=f"Values of k to consider for partial@k advice. Default value: [2,3,...,20]",
    )
    return parser


def main():
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(CWD, "raw_results")
    PLOTS_PATH = os.path.join(CWD, "plots")
    parser = prepare_parser()
    N, r, s, ct, ks = vars(parser.parse_args()).values()
    f_comp = compute_partial_fidelities
    f_plot = plot_partial_fidelities
    # fname = f"fidelity_test_N{N}_reps{r}_step{s}_coaches3_partial_2_20"
    fname = f"fidelity_test_N{N}_reps{r}_step{s}_coaches3_partial_2_20"
    source_name = fname + ".trace"
    res_path = os.path.join(RESULTS_PATH, source_name)
    fig_name = fname + f"_coach_{ct}.pdf"
    fig_path = os.path.join(PLOTS_PATH, fig_name)
    print("Parsing results...")
    traces = parse_traces(res_path)
    print("Magic happens...")
    fidelities = f_comp(traces, coach_type=ct, k_range=ks)
    print("Creating plot...")
    f_plot(fidelities, fig_path, ks)
    print(f"Plot saved at: {fig_path}")


if __name__ == "__main__":
    main()
