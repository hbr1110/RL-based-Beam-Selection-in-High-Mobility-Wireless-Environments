# trainers/train_dqn.py
import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from env.beam_selection_env import BeamSelectionEnv
import glob
import re

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT, 'data')
    all_csv = sorted(glob.glob(os.path.join(DATA_DIR, 'beam_dataset_K*_M*_N*_Ns*_speed*_snr*.csv')))
    MODEL_DIR = os.path.join(ROOT, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 設定 user/stream index（可自動依照欄位數偵測，也可自行調整）
    user_indices = [1, 2, 3, 4]  # 依你的dataset最大K
    stream_indices = [1]         # 目前Ns=1

    for f in all_csv:
        print(f"\n=== 處理 {os.path.basename(f)} ===")
        for user_idx in user_indices:
            for stream_idx in stream_indices:
                print(f"訓練 User {user_idx} Stream {stream_idx}")
                env = BeamSelectionEnv(
                    f,
                    user_idx=user_idx,
                    stream_idx=stream_idx,
                    reward_type='accuracy',
                    max_steps=None,
                    shuffle=True
                )
                check_env(env, warn=True)

                tb_logdir = os.path.join(
                    ROOT, "logs", "dqn_tb", f"{os.path.basename(f).replace('.csv', '')}_user{user_idx}_stream{stream_idx}"
                )

                model = DQN(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log=tb_logdir,
                    learning_rate=1e-3,
                    buffer_size=10000,
                    learning_starts=1000,
                    batch_size=32,
                    gamma=0.99,
                    train_freq=1,
                    target_update_interval=500,
                )
                model.learn(total_timesteps=20000)
                model_name = f'dqn_beam_selection_{os.path.basename(f).replace(".csv", "")}_user{user_idx}_stream{stream_idx}'
                model.save(os.path.join(MODEL_DIR, model_name))
