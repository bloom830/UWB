"""
stft_loss2.py - UWB 信号时频域联合损失函数
功能：
  1. UWBFriendly_Loss: 时域 L1 + 频域 STFT L1 + 峰值保护损失
  2. 短时傅里叶变换 (STFT) 沿时间维度做窗切片，计算谱幅度误差
  3. 峰值损失对脉冲附近的样本额外加权，保护 UWB 的主脉冲形状
数据特征：
  - 输入 [B, 1, 4048]，单通道实数
  - 窗口 win=64, hop=16，得到约 33 个时间帧
  - 通过 time_weight / freq_weight / peak_weight 调整三部分权重
典型使用：
  criterion = UWBFriendly_Loss(win=64, hop=16, time_weight=1.0, freq_weight=0.5, peak_weight=0.2)
"""
import torch
import torch.nn as nn

class UWBFriendly_Loss(nn.Module):
    def __init__(self, win=64, hop=16, time_weight=1.0, freq_weight=0.5, peak_weight=0.2):
        super(UWBFriendly_Loss, self).__init__()
        self.win = win
        self.hop = hop
        self.time_weight = time_weight
        self.freq_weight = freq_weight
        self.peak_weight = peak_weight
        self.l1 = nn.L1Loss()

    def forward(self, est, org):
        """
        est: 模型输出 [B, 1, 4048]
        org: 标签信号 [B, 1, 4048]
        """
        # 1. 时域损失 (MAE)
        loss_time = self.l1(est, org)

        # 2. 频域复数损失
        window = torch.hann_window(self.win).to(est.device)
        stft_est = torch.stft(est.squeeze(1), n_fft=self.win, hop_length=self.hop, 
                              window=window, return_complex=True)
        stft_org = torch.stft(org.squeeze(1), n_fft=self.win, hop_length=self.hop, 
                              window=window, return_complex=True)
        
        # 计算复数频谱差的模长
        loss_freq = torch.mean(torch.abs(stft_est - stft_org))

        # 3. 峰值权重损失 (确保窄脉冲不被平滑)
        # 提取每个样本中脉冲的最大绝对值
        peak_est = torch.max(torch.abs(est), dim=-1)[0]
        peak_org = torch.max(torch.abs(org), dim=-1)[0]
        loss_peak = self.l1(peak_est, peak_org)

        # 总损失
        total_loss = (self.time_weight * loss_time + 
                      self.freq_weight * loss_freq + 
                      self.peak_weight * loss_peak)
        
        return total_loss