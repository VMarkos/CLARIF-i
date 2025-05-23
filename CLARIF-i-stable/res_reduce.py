# res_reduce.py

import os

def reduce_line(line: str) -> str:
    n, steps, _, _, _ = line.split("; ")
    return "; ".join([n, steps, "s", "g", "p"])

def main():
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_PATH = os.path.join(CWD, "raw_results")
    filenames = filter(lambda f: f.endswith(".txt"), os.listdir(RESULTS_PATH))
    for filename in filenames:
        new_filename = filename + ".reduced"
        file_path = os.path.join(RESULTS_PATH, filename)
        new_file_path = os.path.join(RESULTS_PATH, new_filename)
        with open(new_file_path, "w") as new_file:
            new_file.write("")
        with open(file_path, "r") as file:
            for line in file:
                with open(new_file_path, "a") as new_file:
                    new_file.write(f"{reduce_line(line)}\n")

if __name__ == "__main__":
    main()
