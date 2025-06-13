# RL-based Beam Selection in High-Mobility Wireless Environments

本專案針對 高速移動場景的無線通訊系統，實作並比較傳統基準法（Baseline）、隨機選擇（Random）以及深度強化學習（DQN）等 beam selection 策略，探討不同移動速度、SNR、CSI 不完美等現實條件下之效能表現。支援資料產生、模型訓練、批次 sweep、可視化分析。

## 特色
- MATLAB 真實通道模擬：自訂速度/SNR，產生多版本資料集
- 自動化 RL 訓練與 sweep：一鍵訓練多組環境（自動存模型/結果）
- 完美與不完美 CSI 測試：可設定 CSI noise，模擬現實通道估測誤差
- 多策略比較：Baseline（最大 SINR）、Random、DQN Agent 一同比較
- TensorBoard/Notebook：訓練過程/結果可視化，易於分析與簡報
- 專案結構模組化、可擴展性高：便於加入先進 RL、LSTM/PPO 或其他演算法

## 專案結構
- `matlab/`: 資料產生（產生 beam_dataset.csv）
- `data/`: 所有訓練與測試資料
- `utils/`: 資料讀取與驗證
- `env/`: RL 環境接口
- `agents/`: 各種 baseline 或 RL agent
- `trainers/`: 訓練與評估主流程
- `notebooks/`: 資料分析與可視化

## 功能簡介
- 資料集產生
支援多組速度/SNR/CSI noise 設定
欄位說明（user_id, time_idx, sinr_b1N, label, phi_main, aod_path1P）
- RL 環境
OpenAI Gym API
支援 CSI noise，可模擬 channel estimation 不完美
- Baseline/Random/DQN 三種 Agent
Baseline：每步選 SINR 最大（理論上限）
Random：隨機選擇
DQN：深度 Q 學習
- 自動 sweep 與批次比較
train_all.py 批次訓練所有資料集，自動統整結果
notebooks/analysis.ipynb 產出對照圖、表格

## 快速開始
```bash
# 1. 建立虛擬環境，安裝 requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 產生資料 (MATLAB)
cd matlab
執行 generate_beam_dataset.m 產生 beam selection 訓練/測試資料
（可設定不同 speed/SNR/CSI noise）

# 3. 執行 baseline 評估
python train_all.py
# 或單獨訓練（如 baseline/dqn/compare）
python -m trainers.train_baseline
python -m trainers.train_dqn
python -m trainers.train_compare

# 4. 可視化結果
TensorBoard：訓練過程監控
tensorboard --logdir logs/dqn_tb
Jupyter Notebook：統計/畫圖/深入比較
jupyter notebook notebooks/analysis.ipynb




