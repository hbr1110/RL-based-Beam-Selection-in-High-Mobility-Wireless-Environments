# agents/baseline_agent.py

import numpy as np

class BaselineAgent:
    def __init__(self, num_beams):
        self.num_beams = num_beams

    def select_action(self, state):
        # state 可能有多段 history，這裡永遠抓最後一段
        last_sinr = state[-self.num_beams:]
        return int(np.argmax(last_sinr))