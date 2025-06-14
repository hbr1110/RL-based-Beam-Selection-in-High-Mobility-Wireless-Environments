function generate_beam_data_multiuser_mimo(params)
% 完全參照論文：Deep RL for Multi-user Massive MIMO with Channel Aging
% 支援多用戶、多天線、多流，多user/interference SINR

% ===== 參數設定 =====
if ~isfield(params,'K'), params.K = 4; end         % User數
if ~isfield(params,'M'), params.M = 8; end         % BS天線
if ~isfield(params,'N'), params.N = 2; end         % User天線
if ~isfield(params,'Ns'), params.Ns = 1; end       % 每user資料流數
if ~isfield(params,'T'), params.T = 1000; end      % 時間步長
if ~isfield(params,'snrdB'), params.snrdB = 20; end
if ~isfield(params,'delta_t'), params.delta_t = 1e-3; end
if ~isfield(params,'speed'), params.speed = 30; end
if ~isfield(params,'fc'), params.fc = 3.5e9; end
if ~isfield(params,'seed'), params.seed = 42; end
if ~isfield(params,'CB_size'), params.CB_size = 8; end % codebook size (beam數)
if ~isfield(params,'filename'), params.filename = '../data/multiuser_mimo_beam_dataset.csv'; end

rng(params.seed);

K = params.K; M = params.M; N = params.N; Ns = params.Ns; T = params.T;
CB_size = params.CB_size;
noise_power = 1/(10^(params.snrdB/10));
P = 1; % 送一單位能量，SINR只比值，後續可縮放

lambda = 3e8 / params.fc;
fd = (params.speed*1000/3600) / lambda; % max Doppler shift
rho = besselj(0, 2*pi*fd*params.delta_t);

% ==== 建立 Codebook ====
codebook = zeros(M, CB_size);
for b=1:CB_size
    % ULA定向beam, [-pi/2, pi/2] 均勻分布
    angle = -pi/2 + pi*(b-1)/(CB_size-1);
    codebook(:,b) = exp(1j*pi*(0:M-1)'*sin(angle)) / sqrt(M);
end

% ==== 表頭 ====
header = "time_idx";
for k = 1:K
    for n = 1:Ns
        for b = 1:CB_size
            header = header + sprintf(",user%d_stream%d_sinr_b%d",k,n,b);
        end
        header = header + sprintf(",user%d_stream%d_label",k,n);
    end
end

all_data = [];

% ==== 初始化通道 ====
H_cell = cell(K,1);
for k=1:K
    H_cell{k} = (randn(N,M) + 1j*randn(N,M))/sqrt(2);
end

for t=1:T
    % Channel aging (每user)
    for k=1:K
        H_cell{k} = rho*H_cell{k} + sqrt(1-rho^2)*(randn(N,M)+1j*randn(N,M))/sqrt(2);
    end

    data_row = t;
    for k=1:K
        Hk = H_cell{k}; % (N x M)
        for n=1:Ns
            sinr_list = zeros(1, CB_size);
            for b = 1:CB_size
                pk_n = codebook(:,b);          % (M x 1) BS precoder
                wk_n = Hk(:,1);                % (N x 1) 簡單最大比合成 (僅用第一根, 可升級SVD)
                wk_n = wk_n / norm(wk_n);      % 單位化

                % === 信號功率 ===
                signal = sqrt(P/(K*Ns)) * (wk_n') * Hk * pk_n; % (1 x N)*(N x M)*(M x 1) = scalar
                signal_power = abs(signal)^2;

                % === 干擾 ===
                intra = 0; % 若 Ns > 1 可考慮同user不同流干擾

                % Inter-user 干擾: 其他user的所有流, 這裡簡化每流用隨機一個beam
                inter = 0;
                for j=1:K
                    if j == k, continue; end
                    Hj = H_cell{j}; % (N x M)
                    for i=1:Ns
                        bj = randi(CB_size);
                        pj_i = codebook(:,bj); % (M x 1)
                        % 其他user也用最大比合成 (僅用第一根)
                        wj_i = Hj(:,1); wj_i = wj_i/norm(wj_i);
                        % 但干擾投影仍用 k user的combiner (重要！)
                        inter = inter + abs((wk_n')*Hk*pj_i)^2;
                    end
                end
                inter = inter * (P/(K*Ns));

                % === SINR ===
                sinr_list(b) = signal_power / (intra + inter + noise_power*norm(wk_n)^2);
            end
            % === Label: 最大SINR的beam ===
            [~, label] = max(sinr_list);

            % ==== 存這個user/stream所有beam SINR與label ====
            data_row = [data_row, sinr_list, label];
        end
    end
    all_data = [all_data; data_row];
end

% ==== 輸出csv ====
fid = fopen(params.filename, 'w');
fprintf(fid, '%s\n', header);
fclose(fid);
dlmwrite(params.filename, all_data, '-append');
fprintf("已產生多user多流 MIMO SINR 資料集 %s\n", params.filename);

end

