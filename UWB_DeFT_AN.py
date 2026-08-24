"""
UWB_DeFT_AN.py - UWB 去噪主干模型（DeFT-AN 风格）
功能：
  1. DenseBlock: 局部时频特征提取，多层 1D 卷积通道增长
  2. FrequencyAttention / TimeAttention: 双注意力机制
  3. TransformerEncoder: 长程时序依赖建模
  4. UWB_Network: 端到端去噪网络 (输入/输出 [B, 1, 4048])
数据特征：
  - 输入/输出形状：[Batch, 1, 4048]，单通道实数
  - 内部嵌入通道数可配置 (32/64/128)，Transformer 层数可配置
  - 残差连接 + LayerNorm，训练稳定
典型使用：
  model = UWB_Network(in_channels=1, out_channels=1, seq_len=4048)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

# ==================== 1. DenseBlock (局部时频特征提取) ====================
class DenseBlock(nn.Module):
    def __init__(self, in_channels, depth, freq_dim=33):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.freq_dim = freq_dim
        self.block = nn.ModuleList([])
        
        for i in range(self.depth):
            self.block.append(nn.Sequential(
                nn.Conv2d(self.in_channels * (i + 1), self.in_channels, 
                         kernel_size=(3, 3), padding=(1, 1)),
                nn.LayerNorm(self.freq_dim),
                nn.PReLU(in_channels)
            ))
    
    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = self.block[i](skip)
            skip = torch.cat([out, skip], dim=1)
        return out

# ==================== 2. Freq_Conformer (频域全局相关性) ====================
class Freq_Conformer(nn.Module):
    def __init__(self, dim_model, num_head=4, dropout=0.1):
        super(Freq_Conformer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, num_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.GELU(),
            nn.Linear(dim_model * 4, dim_model)
        )
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, x):
        # x: [B*L, F, C]
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# ==================== 3. Time_Conformer (时域序列建模) ====================
class Time_Conformer(nn.Module):
    def __init__(self, dim_model, num_head=4, dropout=0.1, max_len=512):
        super(Time_Conformer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, num_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim_model)
        self.conv = nn.Sequential(
            nn.Conv1d(dim_model, dim_model, kernel_size=3, padding=1, groups=dim_model),
            nn.PReLU(dim_model),
            nn.Conv1d(dim_model, dim_model, kernel_size=1)
        )
        self.norm2 = nn.LayerNorm(dim_model)
        self.pos_enc = PositionalEncoding(dim_model, dropout, max_len)

    def forward(self, x):
        # x: [B*F, L, C]
        x = x + self.pos_enc(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        conv_in = x.transpose(1, 2)
        conv_out = self.conv(conv_in).transpose(1, 2)
        x = self.norm2(x + conv_out)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ==================== 4. Proposed Block (DeFT-AN 核心单元) ====================
class proposed_block(nn.Module):
    def __init__(self, ch_dim, num_head, dropout, depth, freq_dim=33):
        super(proposed_block, self).__init__()
        self.dense = DenseBlock(ch_dim, depth, freq_dim)
        self.freq_conf = Freq_Conformer(ch_dim, num_head, dropout)
        self.time_conf = Time_Conformer(ch_dim, num_head, dropout)

    def forward(self, x):
        B, C, L, F = x.size()
        x = self.dense(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B*L, F, C)
        x = self.freq_conf(x)
        x = x.view(B, L, F, C).permute(0, 2, 1, 3).contiguous().view(B*F, L, C)
        x = self.time_conf(x)
        x = x.view(B, F, L, C).permute(0, 3, 2, 1).contiguous()
        return x

# ==================== 4.5 Noise Estimation Branch ====================
class NoiseEstimationBranch(nn.Module):
    def __init__(self, ch_dim, num_layer=2, freq_dim=33):
        super(NoiseEstimationBranch, self).__init__()
        self.freq_dim = freq_dim
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch_dim, ch_dim, kernel_size=(3, 3), padding=(1, 1)),
                nn.LayerNorm(self.freq_dim),
                nn.PReLU(ch_dim)
            )
            for _ in range(num_layer)
        ])
        self.noise_conv = nn.Sequential(
            nn.Conv2d(ch_dim, 2, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, x):
        feat = x
        for layer in self.layers:
            feat = layer(feat)
        noise_spec = self.noise_conv(feat)
        return noise_spec

# ==================== 5. UWB_Network (主网络) ====================
class UWB_Network(nn.Module):
    def __init__(self, win=128, hop=32, ch_dim=48, num_layer=6, depth=4):
        super(UWB_Network, self).__init__()
        self.win = win
        self.hop = hop
        self.freq_dim = win // 2 + 1
        self.ch_dim = ch_dim

        self.inp_conv = nn.Sequential(
            nn.Conv2d(2, ch_dim, kernel_size=(5, 5), padding=(2, 2)),
            nn.LayerNorm(self.freq_dim),
            nn.PReLU(ch_dim)
        )

        self.layers = nn.ModuleList([
            proposed_block(ch_dim, num_head=4, dropout=0.1, depth=depth, freq_dim=self.freq_dim)
            for _ in range(num_layer)
        ])

        self.noise_branch = NoiseEstimationBranch(ch_dim, num_layer=2, freq_dim=self.freq_dim)

        self.out_conv = nn.Sequential(
            nn.Conv2d(ch_dim, 2, kernel_size=(5, 5), padding=(2, 2))
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None: init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, T = x.shape
        x_sq = x.squeeze(1)

        window = torch.hann_window(self.win).to(x.device)
        stft = torch.stft(x_sq, n_fft=self.win, hop_length=self.hop,
                          window=window, return_complex=True)

        spec = torch.view_as_real(stft)
        spec = spec.permute(0, 3, 2, 1).contiguous()

        feat = self.inp_conv(spec)
        for layer in self.layers:
            feat = layer(feat)

        noise_est = self.noise_branch(feat)

        mask = self.out_conv(feat)

        enhanced_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        enhanced_imag = spec[:, 0] * mask[:, 1] + spec[:, 1] * mask[:, 0]

        enhanced_spec = torch.complex(enhanced_real, enhanced_imag)
        enhanced_spec = enhanced_spec.permute(0, 2, 1).contiguous()

        out = torch.istft(enhanced_spec, n_fft=self.win, hop_length=self.hop,
                          window=window, length=T)

        return out.unsqueeze(1)

# ==================== 测试脚本 ====================
if __name__ == "__main__":
    model = UWB_Network(win=64, hop=16, ch_dim=32)
    test_input = torch.randn(4, 1, 4048) # Batch=4
    test_output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {test_output.shape}")
    
    # 打印参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {params / 1e6:.2f} M")