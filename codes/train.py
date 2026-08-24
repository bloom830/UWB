"""
train.py - UWB 信号去噪主训练脚本
功能：
  1. 从 config.yaml 读取训练超参数 (lr, batch_size, epochs 等)
  2. 调用 data_loader.dataload.create_data_loaders 加载 train/val 数据
  3. 构建 UWB_Network (DeFT-AN 模型) 或自定义模型
  4. 使用 AdamW + ReduceLROnPlateau 训练，带梯度裁剪
  5. 自动保存最佳模型 best_uwb_model.pth、loss 曲线、TensorBoard 事件
数据特征：
  - 输入/输出：[B, 1, 4048] 单通道实数
  - 损失：Composite Loss（时域 + 频域 + 峰值保护
  - 训练日志保存到 training_logs/26-07-07train 目录
典型使用：
  python training/train.py
"""
import sys, os
# 确保项目根目录加入 sys.path，支持从任意目录运行本脚本
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yaml
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv
from torch.utils.tensorboard import SummaryWriter

# 解决OpenMP重复初始化问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置Matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入自定义模块
from data_loader.dataload import create_data_loaders
from models.UWB_DeFT_AN import UWB_Network
from models.stft_loss2 import UWBFriendly_Loss

def train():
    # 1. 加载配置文件 (相对脚本所在目录)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_script_dir, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 正在使用设备: {device} ---")

    # 2. 数据准备 (自动计算训练集均值/方差并保存)
    train_loader, val_loader, _ = create_data_loaders(
        data_root=config['DATA_ROOT'], 
        batch_size=config['BATCH_SIZE'],
        load_test=False
    )

    # 3. 初始化模型 (方案 A)
    model = UWB_Network(
        win=config['MODEL']['win'],
        hop=config['MODEL']['hop'],
        ch_dim=config['MODEL']['ch_dim'],
        num_layer=config['MODEL']['num_layer'],
        depth=config['MODEL']['depth']
    ).to(device)

    # 4. 损失函数与优化器
    criterion = UWBFriendly_Loss(
        win=config['MODEL']['win'],
        hop=config['MODEL']['hop'],
        time_weight=config['LOSS']['time_weight'],
        freq_weight=config['LOSS']['freq_weight'],
        peak_weight=config['LOSS']['peak_weight']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    
    # 学习率动态调整：5轮不下降则减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 5. 训练循环
    os.makedirs(config['LOG_DIR'], exist_ok=True)
    
    # 保存配置参数到txt文件
    config_save_path = os.path.join(config['LOG_DIR'], 'config_params.txt')
    with open(config_save_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("训练配置参数\n")
        f.write("="*60 + "\n\n")
        for key, value in config.items():
            if isinstance(value, dict):
                f.write(f"[{key}]\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")
            f.write("\n")
    print(f"配置参数已保存至: {config_save_path}")
    # 初始化 TensorBoard 日志记录器
    writer = SummaryWriter(log_dir=config['LOG_DIR'])
    best_val_loss = float('inf')
    
    # 用于保存损失曲线的数据
    train_losses = []
    val_losses = []

    for epoch in range(config['EPOCHS']):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['EPOCHS']}")
        for noisy, clean in pbar:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            enhanced = model(noisy)
            
            loss = criterion(enhanced, clean)
            loss.backward()

            # --- 关键：梯度裁剪，防止 UWB 脉冲训练时梯度爆炸 ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['GRAD_CLIP'])
            
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                enhanced = model(noisy)
                v_loss = criterion(enhanced, clean)
                val_loss += v_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 保存损失值
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        print(f"结果: Train Loss = {avg_train_loss:.6f} | Val Loss = {avg_val_loss:.6f}")
        
        # 调整学习率
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config['LOG_DIR'], "best_uwb_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"已保存最佳模型至: {save_path}")

    # 保存损失曲线
    def save_loss_curves(train_losses, val_losses, log_dir):
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失', color='blue')
        plt.plot(val_losses, label='验证损失', color='red')
        plt.xlabel('轮次 (Epoch)')
        plt.ylabel('损失值')
        plt.title('训练与验证损失曲线')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 保存图像
        loss_plot_path = os.path.join(log_dir, 'loss_curve.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存CSV文件
        loss_csv_path = os.path.join(log_dir, 'loss_history.csv')
        with open(loss_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])
            for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
                writer.writerow([i+1, train_loss, val_loss])
        
        return loss_plot_path, loss_csv_path
    
    # 关闭 TensorBoard 记录器
    writer.close()
    
    # 保存损失曲线
    loss_plot, loss_csv = save_loss_curves(train_losses, val_losses, config['LOG_DIR'])
    
    print("\n训练任务全部完成。")
    print(f"TensorBoard 日志已保存至: {config['LOG_DIR']}")
    print(f"损失曲线图像已保存至: {loss_plot}")
    print(f"损失历史数据已保存至: {loss_csv}")
    print("使用命令 'tensorboard --logdir=train_logs' 查看训练曲线")

if __name__ == "__main__":
    train()