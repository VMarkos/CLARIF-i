# classification.py

import os
import json
import itertools as it
from tqdm import tqdm

from classification_utils import generate_classification_test_case, find_classification_inertia_action

def main():
    CWD = os.path.abspath(os.path.dirname(__file__))
    RESULTS_DIR = os.path.join(CWD, 'raw_results')
    ns = [10, 15, 20]
    n_its = 5
    results = dict()
    for n, i in tqdm(it.product(ns, range(n_its)), total=len(ns) * n_its):
        test_case = generate_classification_test_case(n, find_classification_inertia_action, keep_advice_track=True)
        test_case.run()
        results[f"{n}-{i}"] = {
            'report': test_case.report(),
            'advice_track': [ ([str(f[0]) for f in p], a[-1].name) for p, a in test_case.advice_track ],
        }
    res_file = os.path.join(RESULTS_DIR, 'test_classifications.json')
    with open(res_file, 'w') as file:
        json.dump(results, file, indent=2)

if __name__ == "__main__":
    main()
