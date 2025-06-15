# train_all.py
import os
import glob
import numpy as np
import pandas as pd
from env.beam_selection_env import BeamSelectionEnv
from agents.baseline_agent import BaselineAgent
from stable_baselines3 import DQN

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
RESULT_DIR = os.path.join(ROOT, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
dataset_files = sorted(glob.glob(os.path.join(DATA_DIR, "beam_dataset_*.csv")))

# 多種 noise sweep
csi_noise_std_list = [0.0, 0.2, 0.5]
reward_type = "relative"   # 或 "accuracy", "sinr", "sum-rate"

class RandomAgent:
    def __init__(self, num_beams): self.num_beams = num_beams
    def select_action(self, state): return np.random.randint(0, self.num_beams)

def evaluate_agent(env, agent, n_ep=5):
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

import re
def parse_multiuser_csv_columns(csv_file):
    """自動解析user數量與每user/stream的欄位結構"""
    import pandas as pd
    df = pd.read_csv(csv_file, nrows=1)
    cols = list(df.columns)
    users = set()
    beams_per_stream = 0
    pat = re.compile(r"user(\d+)_stream(\d+)_sinr_b(\d+)")
    for col in cols:
        m = pat.match(col)
        if m:
            user_idx = int(m.group(1))
            stream_idx = int(m.group(2))
            beam_idx = int(m.group(3))
            users.add((user_idx, stream_idx))
            beams_per_stream = max(beams_per_stream, beam_idx)
    users = sorted(users)
    return users, beams_per_stream

history_steps = 3  # 用過去3步驟的CSI資料來預測 

summary = []
for delay in [1]:
    for csi_noise_std in csi_noise_std_list:
        for dataset in dataset_files:
            # 解析 user/stream 結構
            users, num_beams = parse_multiuser_csv_columns(dataset)
            print(f"\n==== [Noise={csi_noise_std}] {os.path.basename(dataset)} (user/stream={users}) ====")
            for user_idx, stream_idx in users:
                print(f"  [user={user_idx}, stream={stream_idx}]...", end="", flush=True)
                env = BeamSelectionEnv(
                    dataset,
                    user_idx=user_idx, stream_idx=stream_idx,
                    reward_type=reward_type,
                    shuffle=True,
                    csi_noise_std=csi_noise_std,
                    delay_steps=delay,
                    history_steps=history_steps
                )

                # Baseline/Random agent
                base = BaselineAgent(env.num_beams)
                rand = RandomAgent(env.num_beams)
                base_r, base_acc = evaluate_agent(env, base)
                rand_r, rand_acc = evaluate_agent(env, rand)

                # DQN agent
                noise_tag = f"noise{csi_noise_std}".replace('.', 'p')
                model_tag = f"dqn_{os.path.splitext(os.path.basename(dataset))[0]}_user{user_idx}_stream{stream_idx}_{noise_tag}"
                model_path = os.path.join(MODEL_DIR, model_tag)
                model = DQN(
                    "MlpPolicy", env, verbose=0, learning_rate=1e-3, buffer_size=10000,
                    learning_starts=1000, batch_size=32, gamma=0.99,
                    train_freq=1, target_update_interval=500
                )
                model.learn(total_timesteps=20000)
                model.save(model_path)
                dqn_r, dqn_acc = evaluate_dqn_agent(env, model)
                print(f" Base: {base_acc:.3f} Rand: {rand_acc:.3f} DQN: {dqn_acc:.3f}")

                # 統整結果
                summary.append({
                    "dataset": os.path.basename(dataset),
                    "user": user_idx,
                    "stream": stream_idx,
                    "num_beams": num_beams,
                    "csi_noise_std": csi_noise_std,
                    "delay_steps": delay, 
                    "baseline_acc": base_acc,
                    "random_acc": rand_acc,
                    "dqn_acc": dqn_acc,
                    "baseline_reward": base_r,
                    "random_reward": rand_r,
                    "dqn_reward": dqn_r,
                })

# 輸出 summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(RESULT_DIR, "summary_all_noise_sweep.csv"), index=False)
print("\n已儲存全部結果至 results/summary_all_noise_sweep.csv")
