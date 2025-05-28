# animate.py

import os
import sys

from matplotlib import pyplot as plt

from matplotlib import animation as animation

class SortingAnimator:
    def __init__(self, traces_path: str) -> None:
        self._n: int = 0
        self.traces: dict[tuple[int, int], list[list[int]]] = self.__parse_traces(traces_path)
        self.anim: animation.ArtistAnimation | None = None

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
        self._n = len(trace[0])
        return traces

    def generate(self, key: tuple[int, int]) -> None:
        artists = [] # artists that contain each frame of the animation
        fig, ax = plt.subplots()
        ns = list(range(1, self._n + 1))
        colors = ["tab:blue"] * self._n
        for trace in self.traces[key]:
            container = ax.bar(ns, trace, color=colors)
            artists.append(container) # TODO Shift y axis a bit to show 0 as well
        self.anim = animation.ArtistAnimation(fig=fig, artists=artists, interval=400)
        plt.show()

def main():
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(CWD, "raw_results")
    try:
        input_path = sys.argv[1]
    except IndexError:
        raise Exception("Usage: python[3] animate.py <path_to_trace_file>")
    n = int(input("Enter n: "))
    i = int(input("Enter i: "))
    key = (n, i)
    file_path = os.path.join(RESULTS_PATH, input_path)
    animator = SortingAnimator(file_path)
    animator.generate(key)

if __name__ == "__main__":
    main()
