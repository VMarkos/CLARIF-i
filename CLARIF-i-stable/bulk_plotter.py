# bulk_plotter.py

import itertools as it

from plotter import plot

def bulk_cs_plot():
    N = 20
    reps = 100
    coach_types = ('a', 'p', 'r')
    condition_types = ('full', 'partial')
    name_pairs = it.product(coach_types, condition_types)
    res_types = ('f', 'p', 'pf', 'pp', 'ff', 'fp')
    for (coach, condition), res in zip(name_pairs, res_types):
        figname = f"{coach}_{N}_{reps}_{condition}"
        plot(N, reps, res, figname)

def main():
    bulk_cs_plot()

if __name__ == "__main__":
    main()
