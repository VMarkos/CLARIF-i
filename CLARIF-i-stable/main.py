# main.py
import json

from utils import generate_bubble_sort_test_case

def main():
    results = dict()
    for n in range(2, 4):
        results[n] = []
        for _ in range(2):
            bubble_test = generate_bubble_sort_test_case(n)
            bubble_test.run()
            results[n].append(bubble_test.report())
    with open("test.json", "w") as file:
        json.dump(results, file, indent=2)
    # print(f"Steps: {bubble_test._steps}")

if __name__ == "__main__":
    main()