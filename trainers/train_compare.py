# trainer/train_compare.py

import numpy as np
from env.beam_selection_env import BeamSelectionEnv
from agents.baseline_agent import BaselineAgent
from stable_baselines3 import DQN
import os
import glob

class RandomAgent:
    def __init__(self, num_beams):
        self.num_beams = num_beams
    def select_action(self, state):
        return np.random.randint(0, self.num_beams)

def evaluate_rule_agent(env, agent, n_ep=5):
    rewards, accs = [], []
    for _ in range(n_ep):
        state, info = env.reset()
        ep_reward, ep_acc, steps = 0, 0, 0
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if action == info['label']:
                ep_acc += 1
            steps += 1
            state = next_state
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        accs.append(ep_acc/steps)
    return np.mean(rewards), np.mean(accs)

def evaluate_dqn_agent(env, model, n_ep=5):
    rewards, accs = [], []
    for _ in range(n_ep):
        state, info = env.reset()
        ep_reward, ep_acc, steps = 0, 0, 0
        while True:
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if action == info['label']:
                ep_acc += 1
            steps += 1
            state = next_state
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        accs.append(ep_acc/steps)
    return np.mean(rewards), np.mean(accs)

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT, 'data')
    MODEL_DIR = os.path.join(ROOT, 'models')
    all_csv = sorted(glob.glob(os.path.join(DATA_DIR, 'beam_dataset_speed*_snr*.csv')))

    print("\n==== 批次比較所有資料集 ====")
    for f in all_csv:
        print(f"\n=== {os.path.basename(f)} ===")
        env = BeamSelectionEnv(f, reward_type='accuracy', max_steps=None, shuffle=True)
        agents = {
            "Baseline": BaselineAgent(env.num_beams),
            "Random": RandomAgent(env.num_beams)
        }
        for name, agent in agents.items():
            r, acc = evaluate_rule_agent(env, agent, n_ep=3)
            print(f"{name:10s}: reward={r:.1f}, accuracy={acc:.3f}")

        # 加入 DQN agent
        model_name = f'dqn_beam_selection_{os.path.basename(f).replace(".csv", "")}'
        model_path = os.path.join(MODEL_DIR, model_name + '.zip')
        if os.path.exists(model_path):
            model = DQN.load(model_path, env=env)
            r, acc = evaluate_dqn_agent(env, model, n_ep=3)
            print(f"{'DQN':10s}: reward={r:.1f}, accuracy={acc:.3f}")
        else:
            print("[DQN 未訓練，請先執行 trainers/train_dqn.py]")
