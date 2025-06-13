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
dataset_files = sorted(glob.glob(os.path.join(DATA_DIR, "beam_dataset_speed*_snr*.csv")))

# 可自訂多種 noise 參數
csi_noise_std_list = [0.0, 0.2]  # 可調整 sweep 其他數值

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

summary = []

for csi_noise_std in csi_noise_std_list:
    for dataset in dataset_files:
        print(f"\n==== [Noise={csi_noise_std}] Training on {os.path.basename(dataset)} ====")
        # 1. 建立 noisy 環境
        env = BeamSelectionEnv(
            dataset,
            reward_type='accuracy',
            shuffle=True,
            csi_noise_std=csi_noise_std
        )

        # 2. Baseline/Random agent
        base = BaselineAgent(env.num_beams)
        rand = RandomAgent(env.num_beams)
        base_r, base_acc = evaluate_agent(env, base)
        rand_r, rand_acc = evaluate_agent(env, rand)

        # 3. DQN agent 訓練
        # 模型名稱帶入 noise
        noise_tag = f"noise{csi_noise_std}".replace('.', 'p')
        model_tag = f"dqn_{os.path.splitext(os.path.basename(dataset))[0]}_{noise_tag}"
        model_path = os.path.join(MODEL_DIR, model_tag)
        model = DQN(
            "MlpPolicy", env, verbose=0, learning_rate=1e-3, buffer_size=10000,
            learning_starts=1000, batch_size=32, gamma=0.99,
            train_freq=1, target_update_interval=500
        )
        model.learn(total_timesteps=20000)
        model.save(model_path)

        # 4. DQN 測試
        dqn_r, dqn_acc = evaluate_dqn_agent(env, model)
        print(f"Baseline acc={base_acc:.3f}, Random acc={rand_acc:.3f}, DQN acc={dqn_acc:.3f}")

        # 5. 統整結果
        speed = int(dataset.split("speed")[1].split("_")[0])
        snr = int(dataset.split("snr")[1].split(".")[0])
        summary.append({
            "dataset": os.path.basename(dataset),
            "speed": speed,
            "snr": snr,
            "csi_noise_std": csi_noise_std,
            "baseline_acc": base_acc,
            "random_acc": rand_acc,
            "dqn_acc": dqn_acc,
            "baseline_reward": base_r,
            "random_reward": rand_r,
            "dqn_reward": dqn_r,
        })

# 6. 輸出 summary csv（帶 noise tag，避免覆蓋）
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(RESULT_DIR, "summary_all_noise_sweep.csv"), index=False)
print("\n已儲存全部結果至 results/summary_all_noise_sweep.csv")
