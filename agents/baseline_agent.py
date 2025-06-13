# agents/baseline_agent.py

class BaselineAgent:
    """
    簡單的 Baseline Agent
    - 行為：每次總是選擇 sinr 最大的 beam
    """
    def __init__(self, num_beams):
        self.num_beams = num_beams

    def select_action(self, state):
        return int(state.argmax())
