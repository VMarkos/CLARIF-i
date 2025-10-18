# Edax.py

import os
from LiveFeedbackSubprocess import LiveFeedbackSubprocess

class EdaxAgent():
    def __init__(self, engine_path: str, cwd: str) -> None:
        self.ENGINE_PATH = engine_path
        self.CWD = cwd

    def run(self) -> None:
        p = LiveFeedbackSubprocess(self.ENGINE_PATH, self.CWD)
        print(p.run())

if __name__ == "__main__":
    ENGINE_PATH = "./bin/lEdax-native"
    CWD = os.path.abspath(os.path.dirname(__file__))
    EDAX_CWD = os.path.join(CWD, 'edax', 'edax-reversi-4.6')
    edax = EdaxAgent(ENGINE_PATH, EDAX_CWD)
    edax.run()

