clc;clear;close all;
filenames = {
    '../data/beam_dataset_speed30_snr30.csv', ...
    '../data/beam_dataset_speed120_snr30.csv', ...
    '../data/beam_dataset_speed30_snr10.csv', ...
    '../data/beam_dataset_speed120_snr10.csv'
};

label_jumps = zeros(length(filenames), 1);
sinr_means = zeros(length(filenames), 8); % 假設8 beams

for i = 1:length(filenames)
    T = readtable(filenames{i});
    sinr_cols = contains(T.Properties.VariableNames, 'sinr_b');
    sinr_mat = table2array(T(:, sinr_cols));
    label_vec = T.label;
    
    % Label 跳動頻率
    label_diff = diff(label_vec);
    label_jumps(i) = sum(label_diff ~= 0) / length(label_diff);
    
    % 各beam平均
    sinr_means(i, :) = mean(sinr_mat, 1);
    
    % (可選) label 持續時間 histogram
    % [持續時間, ~] = runlength(label_vec);
end

% 比較 label 跳動頻率
disp('不同資料集 label 跳動頻率:');
for i = 1:length(filenames)
    fprintf('%s: %.3f\n', filenames{i}, label_jumps(i));
end

% 比較不同 SNR 不同速度下的 SINR 平均
figure;
bar(sinr_means');
legend(filenames, 'Interpreter','none');
xlabel('Beam Index');
ylabel('平均 SINR');
title('不同資料集 各 Beam 平均 SINR');

% (可加) plot label_jumps vs speed/snr
% 需自行拆解 filenames 提取 speed/snr
