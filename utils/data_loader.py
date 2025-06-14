# utils/data_loader.py

import pandas as pd
import re

def load_beam_dataset(filepath, user_idx=1, stream_idx=1):
    """
    根據欄位自動抓指定 user/stream 的 sinr/label 欄位
    user_idx/stream_idx: 1-based index
    """
    df = pd.read_csv(filepath)
    # 自動抓 sinr/label 欄位
    sinr_cols = [f"user{user_idx}_stream{stream_idx}_sinr_b{i+1}" for i in range(8)]
    label_col = f"user{user_idx}_stream{stream_idx}_label"
    meta = {
        "num_beams": len(sinr_cols),
        "sinr_cols": sinr_cols,
        "label_col": label_col,
        "user_idx": user_idx,
        "stream_idx": stream_idx
    }
    return df, meta

if __name__ == "__main__":
    # 測試讀取範例
    df, meta = load_beam_dataset('data/beam_dataset_K4_M8_N2_Ns1_speed30_snr10.csv')
    print('前五筆資料:')
    print(df.head())
