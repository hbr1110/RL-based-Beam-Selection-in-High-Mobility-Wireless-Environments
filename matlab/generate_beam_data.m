function generate_beam_data(params)
% generate_beam_data 產生 RL beam selection 所需資料集 (csv格式)
%
%   輸入：
%     params : struct, 包含以下欄位
%         numUsers    - 用戶數量
%         numBeams    - Beam 數
%         numAntennas - 發射天線數
%         numTimeSteps- 時間步長數量
%         speedRange  - 移動速度範圍 [min max] (km/h)
%         fc          - 載波頻率 (Hz)
%         numPaths    - 多徑路徑數
%         K           - Rician factor
%         SNRdB       - 訊號雜訊比 (dB)
%         seed        - 隨機種子
%         filename    - 輸出檔案名稱
%
%   輸出：
%     產生 beam_dataset.csv，欄位說明見下

    % 參數預設值設定
    if ~isfield(params,'numUsers'), params.numUsers = 3; end
    if ~isfield(params,'numBeams'), params.numBeams = 16; end
    if ~isfield(params,'numAntennas'), params.numAntennas = 8; end
    if ~isfield(params,'numTimeSteps'), params.numTimeSteps = 300; end
    if ~isfield(params,'speedRange'), params.speedRange = [50 200]; end
    if ~isfield(params,'fc'), params.fc = 3.5e9; end
    if ~isfield(params,'numPaths'), params.numPaths = 5; end
    if ~isfield(params,'K'), params.K = 5; end
    if ~isfield(params,'SNRdB'), params.SNRdB = 10; end
    if ~isfield(params,'seed'), params.seed = 42; end
    if ~isfield(params,'filename'), params.filename = '../data/beam_dataset.csv'; end

    rng(params.seed); % 保證結果可重現

    % ==== 建立 Uniform Linear Array Beamforming Codebook ====
    theta = linspace(-pi/2, pi/2, params.numBeams); % [-90, +90] 度
    W = zeros(params.numAntennas, params.numBeams);
    for k = 1:params.numBeams
        W(:, k) = exp(1j * pi * (0:params.numAntennas-1)' * sin(theta(k))) ...
                  / sqrt(params.numAntennas);
    end

    % ==== 資料主表 ====
    all_data = [];

    for u = 1:params.numUsers
        % 隨機給每個 user 一個速度與起始主徑方向
        speed = params.speedRange(1) + ...
               (params.speedRange(2)-params.speedRange(1)) * rand();
        phi0 = (rand() - 0.5) * pi; % 起始主徑 AoD [-pi/2, pi/2]

        for t = 1:params.numTimeSteps
            % 模擬主徑 AoD 隨時間線性緩慢漂移
            phi = phi0 + (t-1)*pi/180 * (2*rand-1);

            % ==== 多徑通道產生（含主徑與多個副徑）====
            [H, aod_all] = channel_model( ...
                params.numAntennas, params.numPaths, params.K, ...
                phi, t, speed, params.fc);

            % ==== 計算每個 beam 的 received power (SINR可進階) ====
            sinr_vec = zeros(1, params.numBeams);
            for b = 1:params.numBeams
                h_proj = H * W(:, b); % 天線權重投影
                sinr_vec(b) = norm(h_proj)^2;
            end

            [~, best_idx] = max(sinr_vec);

            % 儲存資料 (user_id, time_idx, sinr_1~N, label, phi_main, aod_path_1~P)
            all_data = [all_data; ...
                        u, t, sinr_vec, best_idx, phi, aod_all];
        end
    end

    % ==== 輸出到 CSV，含欄位說明 ====
    N = params.numBeams;
    header = ['user_id,time_idx', ...
              sprintf(',sinr_b%d',1:N), ...
              ',label,phi_main', ...
              sprintf(',aod_path%d',1:params.numPaths)];
    fid = fopen(params.filename,'w');
    fprintf(fid, '%s\n', header);
    fclose(fid);
    dlmwrite(params.filename, all_data, '-append');
    fprintf('產生完成: %s\n', params.filename);

end