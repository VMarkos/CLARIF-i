# plotter.py
import os
from statistics import mean, stdev

import matplotlib.pyplot as plt

def line_as_dict(line: str) -> dict:
    """ Returns just n, steps, start and goal states. """
    line_split = [ x.strip() for x in line.split(";") ]
    return {
        "n": int(line_split[0]),
        "steps": int(line_split[1]),
        "start_state": line_split[2],
        "goal_state": line_split[3],
    }

def line_plot(paths: list[tuple[str]], figname: str, reps: int=100):
    fig, ax = plt.subplots(figsize=(6,6))
    for path, colour, linestyle, label in paths:
        ns = []
        steps = []
        one_std_interval = [ [], [] ] # first list: mean - std, second list: mean + std
        partial_steps = []
        with open(path, "r") as file:
            for i, line in enumerate(iter(file.readline, "")):
                results = line_as_dict(line)
                partial_steps.append(results["steps"])
                new_n = results["n"] not in ns
                if i > 0 and i % reps == reps - 1: # TODO There is surely a more elegant way to handle this...
                    partial_mean = mean(partial_steps)
                    steps.append(partial_mean)
                    partial_std = stdev(partial_steps)
                    one_std_interval[0].append(partial_mean - partial_std)
                    one_std_interval[1].append(partial_mean + partial_std)
                    partial_steps = []
                    ns.append(results["n"])
    # print(f"ns: {ns}")
    ax.plot(ns, steps, color=colour, linestyle=linestyle, label=label)
    ax.fill_between(ns, *one_std_interval, color=colour, alpha=0.1)
    plt.xticks(ticks=ns)
    plt.xlabel("n")
    plt.ylabel("Coaching Steps")
    plt.title("Coachable Search Learnability")
    ax.grid()
    ax.legend()
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
    is_partial = input("Partial results (y/n): ") == "y"
    paths = [
        (f"b_test_N{N}_reps{reps}.txt", "tab:blue", "solid", "Bubble (no mem)"),
        (f"q_test_N{N}_reps{reps}.txt", "tab:orange", "solid", "Quick (no mem)"),
        (f"b_test_N{N}_reps{reps}_memy.txt", "tab:blue", "dashed", "Bubble (with mem)"),
        (f"q_test_N{N}_reps{reps}_memy.txt", "tab:orange", "dashed", "Quick (with mem)"),
        (f"b_test_N{N}_reps{reps}_memy_longy.txt", "tab:blue", "dotted", "Bubble (with long mem)"),
        (f"q_test_N{N}_reps{reps}_memy_longy.txt", "tab:orange", "dotted", "Quick (with long mem)"),
    ]
    partial_paths = [
        (f"bp_test_N{N}_reps{reps}_memn_longn.txt", "tab:blue", "solid", "Bubble (no mem)"),
        (f"qp_test_N{N}_reps{reps}_memn_longn.txt", "tab:orange", "solid", "Quick (no mem)"),
        (f"bp_test_N{N}_reps{reps}_memy_longn.txt", "tab:blue", "dashed", "Bubble (with mem)"),
        (f"qp_test_N{N}_reps{reps}_memy_longn.txt", "tab:orange", "dashed", "Quick (with mem)"),
        (f"bp_test_N{N}_reps{reps}_memy_longy.txt", "tab:blue", "dotted", "Bubble (with long mem)"),
        (f"qp_test_N{N}_reps{reps}_memy_longy.txt", "tab:orange", "dotted", "Quick (with long mem)"),
    ]
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(CWD, "raw_results")
    reduced_suffix = ".reduced" if reduced else ""
    paths = [ (os.path.join(RESULTS_PATH, t[0] + reduced_suffix), ) + t[1:] for t in paths ]
    partial_paths = [ (os.path.join(RESULTS_PATH, t[0] + reduced_suffix), ) + t[1:] for t in partial_paths ]
    line_plot(paths, figname)

if __name__ == "__main__":
    main()
