# env/beam_selection_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from utils.data_loader import load_beam_dataset

class BeamSelectionEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data_path,
        user_idx=1,
        stream_idx=1,
        reward_type='accuracy',
        max_steps=None,
        shuffle=True,
        csi_noise_std=0.0,
        delay_steps=0,  # 0=oracle, >=1: sample-and-hold
    ):
        super().__init__()
        self.df, self.meta = load_beam_dataset(data_path, user_idx, stream_idx)
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
        self.delay_steps = delay_steps
        self.current_step = 0

    def reset(self, *, seed=None, options=None):
        if self.shuffle:
            self.idx_list = list(range(self.df.shape[0]))
            random.shuffle(self.idx_list)
        else:
            self.idx_list = list(range(self.df.shape[0]))
        self.current_step = 0
        self.obs_queue = []
        for i in range(self.delay_steps + 1):
            idx = min(i, len(self.idx_list) - 1)
            obs = self._get_state_custom(idx)
            self.obs_queue.append(obs)
        obs = self.obs_queue[0]
        info = {}
        return obs, info

    def _get_state_custom(self, idx):
        row_idx = self.idx_list[idx]
        sinr_vec = self.df.loc[row_idx, self.meta['sinr_cols']].to_numpy(dtype=np.float32)
        if self.csi_noise_std > 0:
            noise = np.random.normal(0, self.csi_noise_std, size=sinr_vec.shape)
            sinr_vec = np.clip(sinr_vec + noise, 0, None)
        return sinr_vec

    def step(self, action):
        row_idx = self.idx_list[self.current_step]
        label = int(self.df.loc[row_idx, self.meta['label_col']]) - 1  # 1-based → 0-based
        sinr_vec = self.df.loc[row_idx, self.meta['sinr_cols']].to_numpy(dtype=np.float32)
        sinr_max = sinr_vec[label]
        sinr_sel = sinr_vec[action]

        if self.reward_type == 'accuracy':
            reward = 1.0 if action == label else 0.0
        elif self.reward_type == 'sinr':
            reward = sinr_sel / (sinr_max + 1e-8)
        elif self.reward_type == 'sum-rate':
            reward = np.log2(1 + sinr_sel)
        elif self.reward_type == 'relative':
            if sinr_max == 0:
                reward = 0.0
            else:
                reward = 1 - (sinr_max - sinr_sel) / sinr_max
                # 保證 reward ∈ [0, 1]
                reward = max(0.0, reward)
        else:
            raise ValueError('不支援的 reward_type')

        self.current_step += 1
        terminated = self.current_step >= self.max_steps

        if not terminated:
            next_idx = self.current_step + self.delay_steps
            if next_idx < len(self.idx_list):
                next_obs = self._get_state_custom(next_idx)
            else:
                next_obs = np.zeros(self.state_dim, dtype=np.float32)
            self.obs_queue.pop(0)
            self.obs_queue.append(next_obs)
            obs = self.obs_queue[0]
        else:
            obs = None

        info = {
            'label': label,
            'sinr_optimal': sinr_max,
            'sinr_action': sinr_sel
        }
        return obs, reward, terminated, False, info

    def render(self):
        pass

if __name__ == "__main__":
    import os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(ROOT, 'data', 'beam_dataset_K4_M8_N2_Ns1_speed30_snr10.csv')
    env = BeamSelectionEnv(DATA_PATH, user_idx=1, stream_idx=1, delay_steps=0)
    obs, info = env.reset()
    for _ in range(5):
        action = np.argmax(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f'Action={action}, Reward={reward}, Info={info}')
        if terminated or truncated:
            break
