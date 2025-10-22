# plotter.py

import os
import functools as ft
from statistics import mean, stdev

import matplotlib.pyplot as plt


def line_as_dict(line: str) -> dict:
    """Returns just n, steps, start and goal states."""
    line_split = [x.strip() for x in line.split(";")]
    return {
        "n": int(line_split[0]),
        "steps": int(line_split[1]),
        "start_state": line_split[2],
        "goal_state": line_split[3],
    }


def line_plot(paths: list[tuple[str]], figname: str, reps: int = 100):
    fig, ax = plt.subplots(figsize=(6, 6))
    for path, colour, linestyle, label in paths:
        ns = []
        steps = []
        one_std_interval = [[], []]  # first list: mean - std, second list: mean + std
        partial_steps = []
        trivial_counts = []
        with open(path, "r") as file:
            for i, line in enumerate(iter(file.readline, "")):
                results = line_as_dict(line)
                partial_steps.append(results["steps"])
                new_n = results["n"] not in ns
                if (
                    i > 0 and i % reps == reps - 1
                ):  # TODO There is surely a more elegant way to handle this...
                    non_trivial_partial_steps = []
                    trivial_count = 0
                    for s in partial_steps:
                        if s != -1:
                            non_trivial_partial_steps.append(s)
                        else:
                            trivial_count += 1
                    partial_mean = mean(non_trivial_partial_steps)
                    steps.append(partial_mean)
                    partial_std = stdev(non_trivial_partial_steps)
                    one_std_interval[0].append(partial_mean - partial_std)
                    one_std_interval[1].append(partial_mean + partial_std)
                    partial_steps = []
                    ns.append(results["n"])
                    trivial_counts.append(trivial_count)
        ax.plot(ns, steps, color=colour, linestyle=linestyle, label=label)
        ax.plot(
            ns,
            trivial_counts,
            color=colour,
            marker="*",
            alpha=0.3,
            linestyle=linestyle,
            label=f"{label} (Trivial runs)",
        )
        ax.fill_between(ns, *one_std_interval, color=colour, alpha=0.1)
    plt.xticks(ticks=ns)
    # plt.xlabel("n")
    # plt.ylabel("Coaching Steps")
    plt.tight_layout()
    ax.set_ylim([-10,300])
    # plt.title("Coachable Search Learnability")
    ax.grid()
    ax.legend(loc='upper left')
    CWD = os.path.abspath(os.path.dirname(__file__))
    PLOT_PATH = os.path.join(CWD, "plots")
    if not os.path.isdir(PLOT_PATH):
        os.makedirs(PLOT_PATH)
    fig_path = os.path.join(PLOT_PATH, figname)
    plt.savefig(fig_path + ".pdf")


def main():
    reduced = input("Plotting reduced results (y/n): ") == "y"
    figname = input("Figure filename: ")
    N = int(input("N: "))
    reps = int(input("Repetitions: "))
    res_type = input(
        "Results type ({f}ull, {p}artial, {i}nertia, prepend {p}roactive): "
    )
    plot(N, reps, res_type, figname)

def plot(N, reps, res_type, figname, reduced=False):
    all_paths = {
        "paths": [
            (
                f"b_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:blue",
                "solid",
                "Bubble (no mem)",
            ),
            (
                f"q_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:orange",
                "solid",
                "Quick (no mem)",
            ),
            (
                f"b_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:blue",
                "dashed",
                "Bubble (short mem)",
            ),
            (
                f"q_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:orange",
                "dashed",
                "Quick (short mem)",
            ),
            (
                f"b_test_N{N}_reps{reps}_memy_longy.txt",
                "tab:blue",
                "dotted",
                "Bubble (long mem)",
            ),
            (
                f"q_test_N{N}_reps{reps}_memy_longy.txt",
                "tab:orange",
                "dotted",
                "Quick (long mem)",
            ),
        ],
        "partial_paths": [
            (
                f"bp_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:blue",
                "solid",
                "Bubble (no mem)",
            ),
            (
                f"qp_a_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:orange",
                "solid",
                "Quick (no mem)",
            ),
            (
                f"bp_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:blue",
                "dashed",
                "Bubble (short mem)",
            ),
            (
                f"qp_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:orange",
                "dashed",
                "Quick (short mem)",
            ),
            (
                f"bp_test_N{N}_reps{reps}_memy_longy.txt",
                "tab:blue",
                "dotted",
                "Bubble (long mem)",
            ),
            (
                f"qp_test_N{N}_reps{reps}_memy_longy.txt",
                "tab:orange",
                "dotted",
                "Quick (long mem)",
            ),
        ],
        "proactive_partial_paths": [
            (
                f"bp_p_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:blue",
                "solid",
                "Bubble (no mem)",
            ),
            (
                f"qp_p_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:orange",
                "solid",
                "Quick (no mem)",
            ),
            (
                f"bp_p_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:blue",
                "dashed",
                "Bubble (short mem)",
            ),
            (
                f"qp_p_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:orange",
                "dashed",
                "Quick (short mem)",
            ),
            (
                f"bp_p_test_N{N}_reps{reps}_memy_longy.txt",
                "tab:blue",
                "dotted",
                "Bubble (long mem)",
            ),
            (
                f"qp_p_test_N{N}_reps{reps}_memy_longy.txt",
                "tab:orange",
                "dotted",
                "Quick (long mem)",
            ),
        ],
        "proactive_full_paths": [
            (
                f"b_p_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:blue",
                "solid",
                "Bubble (no mem)",
            ),
            (
                f"q_p_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:orange",
                "solid",
                "Quick (no mem)",
            ),
            (
                f"b_p_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:blue",
                "dashed",
                "Bubble (short mem)",
            ),
            (
                f"q_p_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:orange",
                "dashed",
                "Quick (short mem)",
            ),
            (
                f"b_p_test_N{N}_reps{reps}_memy_longy.txt",
                "tab:blue",
                "dotted",
                "Bubble (long mem)",
            ),
            (
                f"q_p_test_N{N}_reps{reps}_memy_longy.txt",
                "tab:orange",
                "dotted",
                "Quick (long mem)",
            ),
        ],
        "reflexive_partial_paths": [
            (
                f"bp_f_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:blue",
                "solid",
                "Bubble (no mem)",
            ),
            (
                f"qp_f_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:orange",
                "solid",
                "Quick (no mem)",
            ),
            (
                f"bp_f_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:blue",
                "dashed",
                "Bubble (short mem)",
            ),
            (
                f"qp_f_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:orange",
                "dashed",
                "Quick (short mem)",
            ),
            (
                f"bp_f_test_N{N}_reps{reps}_memy_longy.txt",
                "tab:blue",
                "dotted",
                "Bubble (long mem)",
            ),
            (
                f"qp_f_test_N{N}_reps{reps}_memy_longy.txt",
                "tab:orange",
                "dotted",
                "Quick (long mem)",
            ),
        ],
        "reflexive_full_paths": [
            (
                f"b_f_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:blue",
                "solid",
                "Bubble (no mem)",
            ),
            (
                f"q_f_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:orange",
                "solid",
                "Quick (no mem)",
            ),
            (
                f"b_f_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:blue",
                "dashed",
                "Bubble (short mem)",
            ),
            (
                f"q_f_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:orange",
                "dashed",
                "Quick (short mem)",
            ),
            (
                f"b_f_test_N{N}_reps{reps}_memy_longy.txt",
                "tab:blue",
                "dotted",
                "Bubble (long mem)",
            ),
            (
                f"q_f_test_N{N}_reps{reps}_memy_longy.txt",
                "tab:orange",
                "dotted",
                "Quick (long mem)",
            ),
        ],
        "proactive_inertia_paths": [
            (
                f"i_a_test_N{N}_reps{reps}_memn_longn.txt",
                "tab:blue",
                "solid",
                "Inertia (no mem)",
            ),
            (
                f"i_a_test_N{N}_reps{reps}_memy_longn.txt",
                "tab:blue",
                "dashed",
                "Inertia (short mem)",
            ),
            # (f"i_a_test_N{N}_reps{reps}_memy_longy.txt", "tab:blue", "dotted", "Inertia (long mem)"),
        ],
    }
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(CWD, "raw_results")
    reduced_suffix = ".reduced" if reduced else ""
    for k, p in all_paths.items():
        all_paths[k] = [
            (os.path.join(RESULTS_PATH, t[0] + reduced_suffix),) + t[1:] for t in p
        ]
        """
    paths = [ (os.path.join(RESULTS_PATH, t[0] + reduced_suffix), ) + t[1:] for t in paths ]
    partial_paths = [ (os.path.join(RESULTS_PATH, t[0] + reduced_suffix), ) + t[1:] for t in partial_paths ]
    proactive_partial_paths = [ (os.path.join(RESULTS_PATH, t[0] + reduced_suffix), ) + t[1:] for t in proactive_partial_paths ]
    proactive_full_paths = [ (os.path.join(RESULTS_PATH, t[0] + reduced_suffix), ) + t[1:] for t in proactive_full_paths ]
    """
    PATHS = {
        "f": "paths",
        "p": "partial_paths",
        "pi": "proactive_inertia_paths",
        "pp": "proactive_partial_paths",
        "pf": "proactive_full_paths",
        "fp": "reflexive_partial_paths",
        "ff": "reflexive_full_paths",
    }
    line_plot(all_paths[PATHS[res_type]], figname, reps)


if __name__ == "__main__":
    main()
