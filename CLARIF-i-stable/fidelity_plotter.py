# fidelity_plotter.py

import os
import re
import itertools as it

from argparse import ArgumentParser
from copy import deepcopy
from matplotlib import pyplot as plt
from statistics import mean, stdev
from tqdm import tqdm

from utils import (
    compute_fidelity,
    find_bubble_swap_action,
    find_bubble_partial_swap_action,
    find_quick_swap_action,
    find_quick_partial_swap_action,
    find_bubble_swap_action_at_k,
    find_quick_swap_action_at_k,
    batched,
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
PERCENTAGE_CC = dict(zip(PARTIAL_COACH_ACTIONS.keys(), zip(COLORS, MARKERS)))

STYLE_CONFIGS_MAP = lambda c: (
    *CONFIG_COLORS[c[2]],
    "dotted" if c[1] else "dashed" if c[0] else "solid",
)

PARTIAL_STYLE_CONFIGS_MAP = lambda c, k: (
    *PARTIAL_CONFIG_COLORS[(c[2], k)],
    "dotted" if c[1] else "dashed" if c[0] else "solid",
)

PERCENTAGE_SCM = lambda c: (
    *PERCENTAGE_CC[c[2]], 
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
    previous_n = -1
    with open(path, "r") as file:
        for line in iter(file.readline, ""):
            if HEADER_PATTERN.match(line):
                previous_n = n
                n, i = split_header(line)
                if previous_line != None:
                    traces.append(parse_trace_line(previous_line))
                if i == 0:
                    # trace_list.append(deepcopy(traces))
                    if traces != []:
                        trace_list.append( (previous_n, deepcopy(traces)) )
                    traces = []
            previous_line = line
        trace_list.append( (previous_n, deepcopy(traces)) )  # Append last config
    return trace_list

K_STEPS = [4 * 18] * 4 + [3 * 18] * 5 + [2 * 18] * 5 + [1 * 18] * 5

def compute_partial_fidelities(traces: list, coach_type: str="a", k_range=range(2, 21)) -> list:
    fidelities = dict()
    # step = len(traces) // (k_range[-1] - k_range[0])
    # print(step, len(traces), traces[0])
    for k in tqdm(k_range):
        offset = sum(K_STEPS[:(k - 2)])
        # print(offset)
        fidelities[k] = compute_fidelities(traces, coach_type, configs=PARTIAL_CONFIGS, coach_actions=PARTIAL_COACH_ACTIONS, k=k)
    return fidelities

def compute_fidelities(traces: list, coach_type: str="r", configs=CONFIGS, coach_actions=COACH_ACTIONS, k: int=2) -> list:
    """traces as parsed by `parse_traces()`"""
    fidelities = dict()
    offset = sum(K_STEPS[:(k - 2)])
    step = K_STEPS[k - 2]
    chunk_size = step // 18
    # print('->'.join(str(traces[i][0]) for i in (sum(K_STEPS[:j]) for j in range(len(K_STEPS)))))
    grouped_traces = tuple(traces[(offset + i * chunk_size):(offset + (i + 1) * chunk_size)] for i in range(step) )
    # print(offset, step, chunk_size)
    # print(', '.join(map(lambda xs: '-'.join([str(x[0]) for x in xs]), grouped_traces)))
    for ts, config in zip(grouped_traces, configs):
        if (not config[0] and config[1]) or config[3] != coach_type:
            continue
        coach_action_fn = lambda s, ks: coach_actions[config[2]](s, ks, k=k)
        fidelities[config] = {
            n: [compute_fidelity(t, coach_action_fn) for t in trs]
            for n, trs in sorted(ts, key=lambda t: t[0])
        }
    return fidelities

def config_to_str(c: tuple) -> str:
    coach = "Bubble" if c[2][0] == 'b' else "Quick"
    return f"{coach}, Mem: {'Long' if c[1] else 'Short' if c[0] else 'None'}"

def plot_partial_percentage(fidelities: dict, save_path) -> None:
    fig, ax = plt.subplots()
    ax.set_ylim((-0.05, 1.05))
    handles = []
    mean_fs = _get_mean_fs(fidelities)
    for config, mfs in mean_fs.items():
        ps, ms, ss = _aggr_means(mfs)
        c, m, ls = PERCENTAGE_SCM(config)
        ps = [100 * p for p in ps]
        (h,) = ax.plot(ps, ms, label=config_to_str(config), c=c, marker=m, linestyle=ls)
        handles.append(h)
        msp = [ms[i] + ss[i] for i in range(len(ms))]
        msm = [ms[i] - ss[i] for i in range(len(ms))]
        ax.fill_between(ps, msp, msm, color=c, alpha=0.1, linewidth=0.0)
    ax.grid()
    ax.legend(
        handles=sorted(handles, key=lambda h: h._label),
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(save_path)

def _aggr_means(means) -> tuple[list, list]:
    aggr_ms = dict()
    for n, ms in means.items():
        for p, m, s in zip(*ms):
            added = False
            for k in aggr_ms.keys():
                if abs(p - k) < 2e-2:
                    added = True
                    aggr_ms[k][0].append(m)
                    aggr_ms[k][1].append(s)
                    break
            if not added:
                aggr_ms[p] = [[m], [s]]
    vs = zip(*sorted(( (p, mean(ms[0]), mean(ms[1])) for p, ms in aggr_ms.items() ), key=lambda x: x[0]))
    return vs

def _y_offset(ys, dy):
    return [y + dy for y in ys]

def _get_mean_fs(fidelities: dict) -> dict:
    reshaped_fs = _reshape_fidelities(fidelities)
    mean_fs = dict()
    for n, fidelities in reshaped_fs.items():
        for config, fs in fidelities.items():
            if config not in mean_fs.keys():
                mean_fs[config] = dict()
            for k, fidelity in fs.items():
                p = k / n
                m = mean(fidelity)
                s = stdev(fidelity)
                if n not in mean_fs[config].keys():
                    mean_fs[config][n] = [[], [], []]
                mean_fs[config][n][0].append(p)
                mean_fs[config][n][1].append(m)
                mean_fs[config][n][2].append(s)
    return mean_fs
    
def _reshape_fidelities(fidelities: dict, k_range=range(2, 21), n_range=range(5, 21, 5)) -> dict:
    reshaped_fs = dict()
    for k in k_range:
        for config, fs in fidelities[k].items():
            for n, fidelity in fs.items():
                if n not in reshaped_fs.keys():
                    reshaped_fs[n] = dict()
                if config not in reshaped_fs[n].keys():
                    reshaped_fs[n][config] = dict()
                reshaped_fs[n][config][k] = fidelity
    return reshaped_fs


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

def plot_min_max_fidelities(fidelities: dict, save_path, n_range=range(5, 21, 5)) -> None:
    fig, ax = plt.subplots()
    ax.set_ylim((-0.05, 1.05))
    handles = []
    # print(fidelities.keys())
    max_fidelities = _get_max_fidelities(fidelities, n_range)
    labels = ("full", "partial")
    for f, lab in zip((max_fidelities, fidelities[2]), labels):
        for config, fidelity in f.items():
            add_line(config, fidelity, ax, handles, 2 if lab == "partial" else 20, lab)
    ax.grid()
    ax.legend(
        handles=sorted(handles, key=lambda h: h._label),
        loc="lower left",
    )
    fig.tight_layout()
    fig.savefig(save_path)

def _get_max_fidelities(fidelities: dict, n_range) -> dict:
    max_fs = dict()
    for n in n_range:
        for config, fidelity in fidelities[n].items():
            fs = { k: v for k, v in fidelity.items() if k == n}
            if config not in max_fs.keys():
                max_fs[config] = fs
            else:
                max_fs[config].update(fs)
    return max_fs

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

def add_line(config, fidelity, ax, handles, k, lab: str="") -> None:
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
        label=config_to_str(config) + f" ({lab})",
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
    parser.add_argument(
        "-e",
        action="store_true",
        help=f"Boolean flag determining whether to plot only edge cases. Default value: False",
    )
    parser.add_argument(
        "-p",
        action="store_true",
        help=f"Boolean flag determining whether to plot partial fidelities per n. Default value: False",
    )
    return parser


def main():
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(CWD, "raw_results")
    PLOTS_PATH = os.path.join(CWD, "plots")
    parser = prepare_parser()
    N, r, s, ct, ks, edge, pper = vars(parser.parse_args()).values()
    f_comp = compute_partial_fidelities
    if edge:
        f_plot = lambda fs, p: plot_min_max_fidelities(fs, p, range(s, N + 1, s))
    elif pper:
        f_plot = lambda fs, p: plot_partial_percentage(fs, p)
    else:
        f_plot = lambda fs, p: plot_partial_fidelities(fs, p, ks)
    # fname = f"fidelity_test_N{N}_reps{r}_step{s}_coaches3_partial_2_20"
    fname = f"fidelity_test_N{N}_reps{r}_step{s}_coaches3_partial_2_20"
    source_name = fname + ".trace"
    res_path = os.path.join(RESULTS_PATH, source_name)
    fig_name = fname + f"_coach_{ct}.pdf"
    if edge:
        fig_name = "edge_" + fig_name
        ks = range(2, N + 1)
    elif pper:
        fig_name = "pper_" + fig_name
    fig_path = os.path.join(PLOTS_PATH, fig_name)
    print("Parsing results...")
    traces = parse_traces(res_path)
    print("Magic happens...")
    fidelities = f_comp(traces, coach_type=ct, k_range=ks)
    print("Creating plot...")
    f_plot(fidelities, fig_path)
    print(f"Plot saved at: {fig_path}")


if __name__ == "__main__":
    main()
