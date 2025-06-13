# trainers/test_dqn.py

import os
import glob
from stable_baselines3 import DQN
from env.beam_selection_env import BeamSelectionEnv

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT, 'data')
    MODEL_DIR = os.path.join(ROOT, 'models')
    all_csv = sorted(glob.glob(os.path.join(DATA_DIR, 'beam_dataset_speed*_snr*.csv')))

    print("\n==== DQN Agent 批次測試所有資料集 ====")
    for f in all_csv:
        print(f"\n=== 測試 {os.path.basename(f)} ===")
        env = BeamSelectionEnv(f, reward_type='accuracy', max_steps=None, shuffle=True)
        model_name = f'dqn_beam_selection_{os.path.basename(f).replace(".csv", "")}'
        model_path = os.path.join(MODEL_DIR, model_name + '.zip')
        if not os.path.exists(model_path):
            print("[DQN 未訓練，請先執行訓練]")
            continue
        model = DQN.load(model_path, env=env)
        obs, info = env.reset()
        total_reward, total_steps, correct = 0, 0, 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_steps += 1
            if action == info['label']:
                correct += 1
            if terminated or truncated:
                break
        print(f"DQN agent accuracy: {correct/total_steps:.3f}, total_reward: {total_reward}")
