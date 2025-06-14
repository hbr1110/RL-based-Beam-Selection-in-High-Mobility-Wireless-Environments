function analyze_multiuser_beam_dataset(filenames)
    % filenames: cell array, e.g. {'beam_dataset_K4_M8_N2_Ns1_speed30_snr10.csv', ...}

    for fidx = 1:length(filenames)
        filename = filenames{fidx};
        fprintf('=== 分析 %s ===\n', filename);
        T = readtable(filename);

        % 從檔名自動解析 K, Ns, n_beam
        expr = 'K(\d+)_M(\d+)_N(\d+)_Ns(\d+)_speed(\d+)_snr(\d+)';
        token = regexp(filename, expr, 'tokens');
        if isempty(token)
            error('Filename format error');
        end
        K = str2double(token{1}{1});
        Ns = str2double(token{1}{4});
        speed = str2double(token{1}{5});
        snr = str2double(token{1}{6});

        % 動態判斷 beam 數
        varnames = T.Properties.VariableNames;
        tmp = startsWith(varnames, 'user1_stream1_sinr_b');
        n_beam = sum(tmp);

        %% 單一 user/stream: SINR 分布、label 分布、時間變化
        for k = 1:K
            for s = 1:Ns
                prefix = sprintf('user%d_stream%d_', k, s);
                sinr_cols = find(contains(varnames, [prefix 'sinr_b']));
                label_col = find(strcmp(varnames, [prefix 'label']));
                sinr = table2array(T(:, sinr_cols));
                label = table2array(T(:, label_col));

                % 1. 不同 beam SINR 的 histogram
                figure; 
                for b = 1:n_beam
                    subplot(n_beam,1,b); 
                    histogram(sinr(:,b), 40); 
                    title(sprintf('user%d-stream%d SINR of beam%d', k,s,b)); 
                    xlabel('SINR'); ylabel('count');
                end

                % 2. label (最佳beam) 分布
                figure;
                histogram(label, 1:n_beam+1);
                title(sprintf('user%d-stream%d label (最佳beam) 分布',k,s));
                xlabel('beam index'); ylabel('count');

                % 3. label (最佳beam) 隨時間變化
                figure;
                plot(label, 'o-');
                title(sprintf('user%d-stream%d label vs. time',k,s));
                xlabel('time slot'); ylabel('beam label');

                % 4. 所有 SINR 分布 (flatten)
                figure;
                histogram(sinr(:), 80);
                title(sprintf('user%d-stream%d 所有 SINR 分布',k,s));
                xlabel('SINR'); ylabel('count');

                % 5. 每個 beam 的平均 SINR
                figure;
                bar(mean(sinr,1));
                title(sprintf('user%d-stream%d 每beam平均SINR',k,s));
                xlabel('beam index'); ylabel('mean SINR');
            end
        end

        %% 跨檔案存儲：所有 beam 的平均 SINR
        % （以便多個 dataset 跨 speed/snr 畫比較）
        avg_sinr_thisfile = nan(K*Ns, n_beam);
        for k = 1:K
            for s = 1:Ns
                prefix = sprintf('user%d_stream%d_', k, s);
                sinr_cols = find(contains(varnames, [prefix 'sinr_b']));
                sinr = table2array(T(:, sinr_cols));
                avg_sinr_thisfile((k-1)*Ns+s, :) = mean(sinr, 1);
            end
        end
        assignin('base', sprintf('avg_sinr_%s', strrep(filename, '.csv', '')), avg_sinr_thisfile);

    end
end

clc;clear;
close all;
% 呼叫範例
analyze_multiuser_beam_dataset({ ...
                                '../data/beam_dataset_K4_M8_N2_Ns1_speed30_snr10.csv', ...
                                '../data/beam_dataset_K4_M8_N2_Ns1_speed120_snr10.csv', ...
                                '../data/beam_dataset_K4_M8_N2_Ns1_speed30_snr30.csv', ...
                                '../data/beam_dataset_K4_M8_N2_Ns1_speed120_snr30.csv' ...
                                });
