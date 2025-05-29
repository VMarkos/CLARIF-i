# animate.py

import os
import sys
import itertools as it

from matplotlib import pyplot as plt

from matplotlib import animation as animation

class SortingAnimator:
    def __init__(self, traces_path: str, interval: int = 400) -> None:
        self.ALGORITHMS = {
            "b": "Bubble sort",
            "q": "Quick sort",
            "bp": "Bubble sort (partial)",
            "qp": "Quick sort (partial)",
        }
        self.__analyse_path(traces_path)
        self.traces: dict[tuple[int, int], list[list[int]]] = self.__parse_traces(traces_path)
        self.anim: animation.ArtistAnimation | None = None
        self.interval: int = interval

    def __analyse_path(self, path: str) -> tuple[str, int, int, bool, bool]:
        name_split = os.path.basename(path)[:-6].split("_")
        self._algorithm = self.ALGORITHMS[name_split[0]]
        self._n = int(name_split[2][1:])
        self._reps = int(name_split[3][4:])
        self._mem = name_split[4][-1] == "y"
        self._long = name_split[5][-1] == "y"

    def __parse_traces(self, path: str) -> dict[tuple[int, int], list[list[int]]]:
        traces = dict()
        with open(path, "r") as file:
            current_key = None
            for line in file:
                words = [x.strip() for x in line.split(";")]
                is_header = True
                try:
                    _ = int(words[0])
                except ValueError:
                    is_header = False
                if is_header:
                    current_key = tuple(map(int, words))
                    traces[current_key] = []
                else:
                    trace = [ [ int(w.split("=")[-1]) for w in word.split(",") ] for word in words ]
                    trace = [ t for i, t in enumerate(trace) if t not in trace[:i] ]
                    traces[current_key] = trace # NOTE This skips intentionally everything except for the last iteration; a more efficient way must exist
        return traces

    def generate(self, key: tuple[int, int]) -> None:
        artists = [] # artists that contain each frame of the animation
        fig, ax = plt.subplots()
        plt.suptitle(f"{self._algorithm}: N={self._n}, iteration {key[1] + 1} / {self._reps}", y=0.99, fontsize=18)
        plt.title(f"With{'' if self._mem else 'out'} Memory; {'' if self._long else 'Not'} Long", fontsize=10)
        ns = list(range(1, self._n + 1))
        traces = self.traces[key]
        differences = ( [ i for i, (a, b) in enumerate(zip(x, y)) if a != b ] for x, y in zip(traces[:-1], traces[1:]) )
        for trace, diff in it.zip_longest(traces, differences, fillvalue=[]):
            colors = [ "tab:orange" if i in diff else "tab:blue" for i in range(self._n) ] 
            container = ax.bar(ns, [ x + 1 for x in trace ], color=colors)
            artists.append(container) # TODO Shift y axis a bit to show 0 as well
        self.anim = animation.ArtistAnimation(fig=fig, artists=artists, interval=self.interval)

    def save(self, path: str) -> None:
        """ Assuming that `path` corresponds to a PillowWriter valid extension (.gif, .apng, .webp) """
        self.anim.save(path, writer="pillow")

def main():
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(CWD, "raw_results")
    ANIMATIONS_PATH = os.path.join(CWD, "animations")
    if not os.path.isdir(ANIMATIONS_PATH):
        os.mkdir(ANIMATIONS_PATH)
    try:
        input_path = sys.argv[1]
    except IndexError:
        raise Exception("Usage: python[3] animate.py <path_to_trace_file>")
    n = int(input("Enter n: "))
    i = int(input("Enter i: "))
    interval = int(int_str) if (int_str := input("Enter interval: ")) != "" else 400
    key = (n, i)
    file_path = os.path.join(RESULTS_PATH, input_path)
    animator = SortingAnimator(file_path, interval=interval)
    print("Generating animation...")
    animator.generate(key)
    save_path = os.path.join(ANIMATIONS_PATH, input_path[:-6] + "_" + "_".join(map(str, key)) + ".gif")
    animator.save(save_path)
    print(f"Saved animation at: {save_path}")

if __name__ == "__main__":
    main()
