# trainers/train_dqn.py

import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from env.beam_selection_env import BeamSelectionEnv
import glob

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT, 'data')
    all_csv = sorted(glob.glob(os.path.join(DATA_DIR, 'beam_dataset_speed*_snr*.csv')))
    MODEL_DIR = os.path.join(ROOT, 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)

    for f in all_csv:
        print(f"\n=== 訓練 DQN for {os.path.basename(f)} ===")
        env = BeamSelectionEnv(f, reward_type='accuracy', max_steps=None, shuffle=True)
        check_env(env, warn=True)

        tb_logdir = os.path.join(ROOT, "logs", "dqn_tb", os.path.basename(f).replace('.csv', ''))

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
        model_name = f'dqn_beam_selection_{os.path.basename(f).replace(".csv", "")}'
        model.save(os.path.join(MODEL_DIR, model_name))
