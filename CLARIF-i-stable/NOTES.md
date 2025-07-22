# Implementation Notes

Implementation notes about CLARIF-i development and, mostly, any experiments conducted.

## Bugs Log

1. Major bug: Rules take up a lot of memory in the case of complete states as conditions (`O(n!)`), so we should better generate rules dynamically.
2. Minor bug: Merge sort relies on insertion and not swapping, so quick sort might be a better idea to compare against bubble sort.
3. Major bug: Even when dynamically generating rules, things take a lot of time, since in each case we run the corresponding algorithm from scratch to compute the rule (maybe cache them? have some sort of memory?)

## Next Steps

### Major Steps

- [x] Implement two sorting algorithms with different worst case complexities to demonstrate learnability. **Done! Implemented Bubble and Quick Sort(s).**
- [x] Add memory to learners so as to measure the effect of coaching on proactively catching future cases.
- [x] As a sub-objective of the above, first implement rules that have partial states as conditions in the cases of Bubble and Quick Sorts.
- [x] Create a `plotter.py` to create plots of various kinds needed for the cases above.

### Minor Steps / Bug Fixes

- [x] Compute averages in line plots.
- [x] Add min-max or **+- std buffer** in line plots.
- [x] Change `line_plot` to accept a list of paths and plot all plots in one single plot.
- [x] Further extend `line_plot` to show a plot title and other aesthetics (**grid lines, ticks, titles, labels, legend**).
- [x] Bubble sort + mem: Run again `reps=100` experiments for `N=19,20` and append them to the corresponding file to create full plots.
- [ ] 
 
## Considerations

1. Regarding partial states as conditions, in the case of Quick Sort, which is a top-down algorithm, how is it possible to ignore the rest of the state? Also, what about the pivot? (e.g., in our case, where we implement Hoare's two-index approach).
2. With full states and memory things take up a lot of time...
