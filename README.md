# UWB 信号去噪 — DeFT-AN 模型

本项目实现一种基于 **DeFT-AN** 架构的超宽带（UWB）信号去噪模型，结合时频域联合损失函数，在多种信噪比条件下有效恢复 UWB 脉冲信号。

## 项目结构

```
UWB/
├── models/
│   ├── UWB_DeFT_AN.py      # 主干网络（DenseBlock + 双注意力 + Transformer）
│   ├── stft_loss2.py        # 时频域联合损失函数（UWBFriendly_Loss）
│   └── __init__.py
├── training/
│   ├── train.py             # 训练脚本
│   └── config.yaml          # 训练超参数配置
├── testing/
│   ├── test.py              # 通用测试脚本
│   ├── batch_test.py        # 多信噪比批量测试脚本（-15 ~ +15 dB）
│   └── test.yaml            # 测试配置
├── data_loader/             # 数据加载模块
├── data/                    # 数据集（需自行准备）
└── README.md
```

## 主要特点

- **模型**：`UWB_Network` — 输入 `[B, 1, 4048]`，输出同尺寸，参数量约 0.5~2 M（可配置）。
- **损失函数**：`UWBFriendly_Loss` — 融合时域 MAE、频域 STFT 幅度误差和峰值保护项，有效保留 UWB 窄脉冲形状。
- **训练**：自动保存最佳模型、TensorBoard 日志、损失曲线。
- **测试**：支持单 SNR 和多 SNR 批量测试，输出 MSE、SNR 提升、峰值误差、SSIM 等指标。

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy matplotlib pyyaml tqdm scikit-image scipy
```

### 2. 准备数据

将 UWB 数据集放入 `data/` 目录，格式参见 `data_loader/dataload.py`。

### 3. 训练模型

```bash
cd training
python train.py
```

训练日志保存在 `training_logs/` 下。

### 4. 测试模型

```bash
cd testing
python test.py               # 通用测试
python batch_test.py         # 多 SNR 批量测试
```

## 引用

若您使用了本项目的代码或思路，欢迎引用或标注来源。
