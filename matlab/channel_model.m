% =========================================================================
% 子函式：動態多徑 Rician 通道產生
% =========================================================================
function [H, aod_all] = channel_model(Nt, Np, K, phi_main, t, speed, fc)
% channel_model 建立多徑 Rician 通道
%
% 輸入:
%   Nt      - 天線數
%   Np      - 多徑數
%   K       - Rician 因子
%   phi_main- 主徑 AoD (弧度)
%   t       - 時間 index
%   speed   - 用戶速度 (km/h)
%   fc      - 載波頻率 (Hz)
%
% 輸出:
%   H       - 1xNt 複數通道向量
%   aod_all - 1xNp 多徑 AoD (弧度)

    lambda = 3e8 / fc;
    aod_all = zeros(1, Np);
    H = zeros(1, Nt);

    for p = 1:Np
        if p==1
            phi = phi_main; % 主徑方向
        else
            phi = phi_main + (rand-0.5)*pi/6; % 鄰近主徑
        end
        aod_all(p) = phi;
        gain = (randn + 1j*randn)/sqrt(2*Np); % 隨機多徑增益
        fd = (speed * 1000 / 3600) / lambda;
        doppler = exp(1j*2*pi*fd*t*cos(phi)); % Doppler 相位
        if p==1
            gain = sqrt(K/(K+1))*doppler + sqrt(1/(K+1))*gain; % Rician 合成
        end
        a_tx = exp(1j * pi * (0:Nt-1)' * sin(phi)) / sqrt(Nt); % array response
        H = H + gain * a_tx.';
    end
end
