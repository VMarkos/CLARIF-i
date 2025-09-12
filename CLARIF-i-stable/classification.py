# classification.py

import json
from classification_utils import generate_classification_test_case, find_classification_inertia_action

def main():
    n = 10
    test_case = generate_classification_test_case(n, find_classification_inertia_action)
    test_case.run()
    with open('test_classification.json') as file:
        json.dump(test_case.report(), file, indent=2)

if __name__ == "__main__":
    main()
