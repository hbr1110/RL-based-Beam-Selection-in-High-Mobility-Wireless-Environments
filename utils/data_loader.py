# utils/data_loader.py

"""
讀取並檢查 MATLAB 輸出之 beam selection 資料集

- 支援 pandas 讀取與 numpy 轉換
- 自動辨識 beam 數、label 欄位
- 提供資料檢查與簡易統計（label 分布、SINR 範圍等）
"""

import pandas as pd
import numpy as np
import os

def load_beam_dataset(filepath):
    """
    讀取 beam selection 資料
    Args:
        filepath (str): csv 檔案路徑
    Returns:
        df (pd.DataFrame): 資料表
        meta (dict): beam 數、欄位資訊
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'找不到資料集: {filepath}')
    df = pd.read_csv(filepath)
    print(f'共讀入 {df.shape[0]} 筆資料, 欄位數 {df.shape[1]}')

    # 自動抓 beam/sinr 欄位
    sinr_cols = [c for c in df.columns if c.startswith('sinr_b')]
    num_beams = len(sinr_cols)
    print(f'偵測到 beam 數: {num_beams}')
    if num_beams == 0:
        raise ValueError('未偵測到任何 sinr_b 開頭欄位！')

    # label、主徑、所有 AoD 欄位
    label_col = 'label'
    phi_main_col = 'phi_main'
    aod_cols = [c for c in df.columns if c.startswith('aod_path')]

    # 資料檢查
    print('Label 分布:')
    print(df[label_col].value_counts().sort_index())

    # SINR 檢查（每一個 beam 統計）
    print('SINR 各 beam 最大/最小/平均:')
    print(df[sinr_cols].agg(['min', 'max', 'mean']))

    meta = {
        'sinr_cols': sinr_cols,
        'num_beams': num_beams,
        'label_col': label_col,
        'phi_main_col': phi_main_col,
        'aod_cols': aod_cols,
        'columns': df.columns.tolist(),
        'num_rows': df.shape[0]
    }
    return df, meta

if __name__ == "__main__":
    # 測試讀取範例
    df, meta = load_beam_dataset('data/beam_dataset_speed60_snr20.csv')
    print('前五筆資料:')
    print(df.head())
