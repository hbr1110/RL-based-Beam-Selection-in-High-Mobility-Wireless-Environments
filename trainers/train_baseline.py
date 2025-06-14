# trainers/train_baseline.py

import numpy as np
from env.beam_selection_env import BeamSelectionEnv
from agents.baseline_agent import BaselineAgent
import os
import glob

def evaluate_agent(env, agent, num_episodes=10):
    total_rewards = []
    total_accuracy = []

    for ep in range(num_episodes):
        state, info = env.reset()
        ep_rewards = 0
        ep_correct = 0
        steps = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_rewards += reward
            steps += 1
            if action == info['label']:
                ep_correct += 1
            state = next_state
            if terminated or truncated:
                break
        total_rewards.append(ep_rewards)
        total_accuracy.append(ep_correct / steps)
    return np.mean(total_rewards), np.mean(total_accuracy)

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT, 'data')
    all_csv = sorted(glob.glob(os.path.join(DATA_DIR, 'beam_dataset_K*.csv')))

    print("\n==== Baseline Agent 在所有資料集的表現 ====")
    for f in all_csv:
        print(f"\n=== {os.path.basename(f)} ===")
        for user_idx in [1, 2, 3, 4]:  # 依資料集 user 數而定
            env = BeamSelectionEnv(
                f,
                user_idx=user_idx,
                stream_idx=1,  # 若有多 stream 可for迴圈
                reward_type='relative',
                max_steps=None,
                shuffle=True,
                csi_noise_std=0.2,
                delay_steps=100
            )
            agent = BaselineAgent(env.num_beams)
            r, acc = evaluate_agent(env, agent, num_episodes=5)
            print(f"[user{user_idx}] relative reward: {r:.1f}, 平均 accuracy: {acc:.3f}")
