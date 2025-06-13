# env/beam_selection_env.py

"""
Beam Selection RL 環境 (以 OpenAI Gym API 設計)
- 每次 step 給定一個 sinr 向量作為 state
- action = beam index (0 ~ N-1)
- reward: 命中正確 beam (+1), 否則 0 或給 SINR 增益
- 支援 episode 循環、隨機起始
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from utils.data_loader import load_beam_dataset

class BeamSelectionEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, data_path, reward_type='accuracy', max_steps=None, shuffle=True, csi_noise_std=0.2):
        super().__init__()
        self.df, self.meta = load_beam_dataset(data_path)
        self.num_beams = self.meta['num_beams']
        self.state_dim = self.num_beams

        self.action_space = spaces.Discrete(self.num_beams)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.reward_type = reward_type
        self.max_steps = max_steps or self.df.shape[0]
        self.shuffle = shuffle
        self.csi_noise_std = csi_noise_std
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        if self.shuffle:
            self.idx_list = list(range(self.df.shape[0]))
            random.shuffle(self.idx_list)
        else:
            self.idx_list = list(range(self.df.shape[0]))
        self.current_step = 0
        obs = self._get_state()
        info = {}
        return obs, info

    def _get_state(self):
        row_idx = self.idx_list[self.current_step]
        sinr_vec = self.df.loc[row_idx, self.meta['sinr_cols']].to_numpy(dtype=np.float32)
        if self.csi_noise_std > 0:
            # 加入高斯噪聲，模擬估測不完美
            noise = np.random.normal(0, self.csi_noise_std, size=sinr_vec.shape)
            sinr_vec = np.clip(sinr_vec + noise, 0, None)  # SINR 不得小於 0
        return sinr_vec

    def step(self, action):
        row_idx = self.idx_list[self.current_step]
        label = int(self.df.loc[row_idx, self.meta['label_col']]) - 1  # MATLAB 是 1-based!
        sinr_vec = self.df.loc[row_idx, self.meta['sinr_cols']].to_numpy(dtype=np.float32)

        if self.reward_type == 'accuracy':
            reward = 1.0 if action == label else 0.0
        elif self.reward_type == 'sinr':
            reward = sinr_vec[action] / (sinr_vec[label] + 1e-8)
        else:
            raise ValueError('不支援的 reward_type')

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False  # 你如有 early stop 可自定義
        obs = self._get_state() if not terminated else None
        info = {
            'label': label,
            'sinr_optimal': sinr_vec[label],
            'sinr_action': sinr_vec[action]
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass  # 可選

if __name__ == "__main__":
    import os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(ROOT, 'data', 'beam_dataset_speed60_snr20.csv')
    env = BeamSelectionEnv(DATA_PATH)
    obs, info = env.reset()
    for _ in range(5):
        action = np.argmax(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f'Action={action}, Reward={reward}, Info={info}')
        if terminated or truncated:
            break
