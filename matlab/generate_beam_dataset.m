% =========================================================================
%  File Name   : generate_beam_dataset.m
%  Description : 產生適用於 RL beam selection 的多用戶/多時刻/多徑/高速移動
%                通道資料集，支援均勻 beam label 分布，輸出可直接匯入
%                Python 或其他 ML 訓練流程。
%  Author      : [Your Name]
%  Date        : [2025/06/13]
%  Copyright   : (C) 2025. All rights reserved.
%  Revision    : v1.0 初版
% =========================================================================

clc;clear;close all;

speeds = [30, 60, 120]; % 不同移動速度
snrs = [10, 20, 30];    % 不同 SNR
for i = 1:length(speeds)
    for j = 1:length(snrs)
        params = struct( ...
            'numUsers', 4, ...
            'numBeams', 16, ...
            'numAntennas', 8, ...
            'numTimeSteps', 300, ...
            'speedRange', [speeds(i), speeds(i)], ... % 單一速度
            'fc', 3.5e9, ...
            'numPaths', 5, ...
            'K', 5, ...
            'SNRdB', snrs(j), ...
            'seed', 42 + i*10 + j, ...
            'filename', sprintf('../data/beam_dataset_speed%d_snr%d.csv', speeds(i), snrs(j)) ...
        );
        generate_beam_data(params);
    end
end



% ===== beam_dataset.csv 每一筆資料的意義 =====
% 欄位名稱	意義/物理解釋
% user_id	用戶編號（支援多個獨立移動用戶並行產生數據）
% time_idx	時間序號（每筆資料為一個離散時間點，模擬時序序列）
% sinr_b1 ... sinr_bN	每個 beam 編碼簿方向下對應的接收功率（或等效 SINR，與 beam index 一一對應，N 為 beam 數）
% label	在此時間點下，所有 beam 中接收功率最大的 beam index，即「最佳 beam」，是 RL/ML 的學習目標
% phi_main	當下主徑（LOS）入射角（AoD, Angle of Departure, 單位：弧度）
% aod_path1~aod_pathP	每一個多徑分量的入射角（包含主徑與多個非直射徑，P 為多徑路徑數），反映物理通道狀態