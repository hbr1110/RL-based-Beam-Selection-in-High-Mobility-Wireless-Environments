# agents/baseline_agent.py

import numpy as np

class BaselineAgent:
    def __init__(self, num_beams):
        self.num_beams = num_beams

    def select_action(self, state):
        return int(np.argmax(state))  # always select max-SINR beam
