"""
test.py - UWB 通用去噪测试脚本
功能：
  1. 从 test.yaml 读取测试配置 (模型路径、测试集路径、输出目录等)
  2. 加载训练好的模型 (best_uwb_model.pth) 并设为 eval 模式
  3. 在测试集上推理，计算输出与干净信号的 MSE / SNR 改善量
  4. 保存去噪结果 (denoised.pt)、测试指标文本、波形对比图
数据特征：
  - 输入 [B, 1, 4048] 单通道实数
  - 输出同形状；使用 data_loader.dataload.UWBDataset 或 TestDataset
  - 评估关注脉冲峰值保留与整体 MSE 下降
典型使用：
  python testing/test.py
"""
import sys, os
# 确保项目根目录加入 sys.path，支持从任意目录运行本脚本
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
import yaml
from data_loader.dataload import UWBDataset, create_data_loaders
from models.UWB_DeFT_AN import UWB_Network
from skimage.metrics import structural_similarity as ssim  # 导入SSIM

# 解决OpenMP重复初始化问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 确保使用单线程处理，减少内存开销
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. SI-SNR 计算函数 (手动实现) ---
def calculate_si_snr(target, prediction, eps=1e-8):
    """
    计算 Scale-Invariant Signal-to-Noise Ratio (SI-SNR)
    Args:
        target: 纯净信号 (干净的 UWB 脉冲)
        prediction: 预测信号 (模型输出的增强信号)
    Returns:
        si_snr: 标量值
    """
    target = torch.tensor(target) if not isinstance(target, torch.Tensor) else target
    prediction = torch.tensor(prediction) if not isinstance(prediction, torch.Tensor) else prediction
    
    target = target - torch.mean(target)
    prediction = prediction - torch.mean(prediction)

    s_target = torch.sum(target * prediction) * target / (torch.norm(target) ** 2 + eps)
    e_noise = prediction - s_target

    si_snr = 10 * torch.log10((torch.norm(s_target) ** 2 + eps) / (torch.norm(e_noise) ** 2 + eps))
    return si_snr.item()

# --- 2. PSNR 计算函数 ---
def calculate_psnr(target, prediction, data_range=1.0):
    """计算峰值信噪比"""
    mse = np.mean((target - prediction) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return psnr

# --- 3. 新增指标计算函数 ---
def calculate_mse(target, prediction):
    """计算均方误差"""
    return np.mean((target - prediction) ** 2)

def calculate_correlation(target, prediction):
    """计算互相关系数"""
    # 计算归一化后的互相关系数
    target_norm = target - np.mean(target)
    prediction_norm = prediction - np.mean(prediction)
    correlation = np.dot(target_norm, prediction_norm) / (np.linalg.norm(target_norm) * np.linalg.norm(prediction_norm))
    return correlation

def calculate_peak_error(target, prediction):
    """计算峰值位置误差"""
    # 确保目标信号和预测信号长度相同
    if len(target) != len(prediction):
        print(f"警告: 信号长度不匹配 - 目标: {len(target)}, 预测: {len(prediction)}")
        return -1
    
    # 找到目标信号和预测信号的所有峰值位置
    # 使用更精确的峰值检测：寻找超过阈值的局部最大值
    threshold = 0.5 * np.max(np.abs(target))  # 设置阈值为最大幅度的50%
    
    # 找到目标信号的所有峰值位置
    target_peaks = []
    for i in range(1, len(target)-1):
        if np.abs(target[i]) > threshold and np.abs(target[i]) > np.abs(target[i-1]) and np.abs(target[i]) > np.abs(target[i+1]):
            target_peaks.append((np.abs(target[i]), i))
    
    # 找到预测信号的所有峰值位置
    prediction_peaks = []
    for i in range(1, len(prediction)-1):
        if np.abs(prediction[i]) > threshold and np.abs(prediction[i]) > np.abs(prediction[i-1]) and np.abs(prediction[i]) > np.abs(prediction[i+1]):
            prediction_peaks.append((np.abs(prediction[i]), i))
    
    # 如果没有找到峰值，使用原始方法
    if not target_peaks or not prediction_peaks:
        peak_target = np.argmax(np.abs(target))
        peak_prediction = np.argmax(np.abs(prediction))
        
        # 后处理：考虑脉冲重复间隔（125个采样点）
        # 计算相对于原始峰值位置的最小循环误差
        pri_samples = 125  # 脉冲重复间隔的采样点数
        min_error = float('inf')
        for k in range(-10, 11):  # 考虑前后10个PRI
            shifted_pred_pos = peak_prediction - k * pri_samples
            if 0 <= shifted_pred_pos < len(prediction):
                error = np.abs(peak_target - shifted_pred_pos)
                if error < min_error:
                    min_error = error
        
        return min_error
    
    # 按峰值幅度排序
    target_peaks.sort(reverse=True, key=lambda x: x[0])
    prediction_peaks.sort(reverse=True, key=lambda x: x[0])
    
    # 找到最接近的峰值对，考虑脉冲重复间隔
    min_error = float('inf')
    pri_samples = 125  # 脉冲重复间隔的采样点数
    
    for target_amp, target_pos in target_peaks[:3]:  # 只考虑前3个最强峰值
        for pred_amp, pred_pos in prediction_peaks[:3]:  # 只考虑前3个最强峰值
            # 计算相对于目标位置的最小循环误差
            for k in range(-10, 11):  # 考虑前后10个PRI
                shifted_pred_pos = pred_pos - k * pri_samples
                if 0 <= shifted_pred_pos < len(prediction):
                    error = np.abs(target_pos - shifted_pred_pos)
                    if error < min_error:
                        min_error = error
    
    return min_error

def calculate_ssim(target, prediction):
    """计算结构相似性"""
    # 计算SSIM
    try:
        s = ssim(target, prediction, data_range=target.max()-target.min())
        return s
    except:
        return 0.0

# --- 4. 主测试流程 ---
def main():
    # --- 加载配置文件 (相对脚本所在目录) ---
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _config_path = os.path.join(os.path.dirname(_script_dir), "training", "config.yaml")
    with open(_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # --- 配置参数 ---
    DATA_ROOT = config['DATA_ROOT']  # 数据集根目录
    MODEL_PATH = os.path.join(config['LOG_DIR'], "best_uwb_model.pth")  # 训练好的模型路径
    OUTPUT_DIR = os.path.join(os.getcwd(), "test_results") # 结果保存目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    BATCH_SIZE = 8  # 减少批处理大小以降低内存使用
    DEVICE = torch.device("cpu")  # 强制在CPU上运行，避免GPU内存问题

    # --- 初始化模型 ---
    model = UWB_Network(
        win=config['MODEL']['win'],
        hop=config['MODEL']['hop'],
        ch_dim=config['MODEL']['ch_dim'],
        num_layer=config['MODEL']['num_layer'],
        depth=config['MODEL']['depth']
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"模型加载完毕: {MODEL_PATH}")

    # --- 加载测试集 ---
    # 注意: 这里我们不使用 create_data_loaders 返回的 DataLoader 列表, 
    # 而是直接创建测试集以获取索引和原始 SNR 信息
    test_dataset = UWBDataset(DATA_ROOT, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 加载 SNR 信息 (假设 snr_info.json 在数据集目录下) ---
    snr_info_path = os.path.join(DATA_ROOT, "snr_info.json")
    if os.path.exists(snr_info_path):
        import json
        with open(snr_info_path, 'r') as f:
            snr_info = json.load(f)
        test_snr_list = snr_info['test_snr']
    else:
        print("警告: 未找到 snr_info.json, 将使用随机 SNR 分组进行演示。")
        test_snr_list = np.random.uniform(-15, 10, len(test_dataset))

    # --- 推理与指标计算 ---
    results = []
    snr_ranges = {"Low (-15~-5dB)": [], "Medium (-5~5dB)": [], "High (5~15dB)": []}

    with torch.no_grad():
        for batch_idx, (noisy, clean) in enumerate(tqdm(test_loader, desc="测试中")):
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            enhanced = model(noisy).cpu().numpy()
            noisy_np = noisy.cpu().numpy()
            clean_np = clean.cpu().numpy()

            for i in range(enhanced.shape[0]):
                idx_in_batch = batch_idx * BATCH_SIZE + i
                if idx_in_batch >= len(test_dataset):
                    break
                
                # 获取当前样本的 SNR
                snr = test_snr_list[idx_in_batch]
                # 计算指标
                psnr_val = calculate_psnr(clean_np[i, 0], enhanced[i, 0])
                si_snr_val = calculate_si_snr(torch.tensor(clean_np[i, 0]), torch.tensor(enhanced[i, 0]))
                mse_val = calculate_mse(clean_np[i, 0], enhanced[i, 0])
                correlation_val = calculate_correlation(clean_np[i, 0], enhanced[i, 0])
                peak_error_val = calculate_peak_error(clean_np[i, 0], enhanced[i, 0])
                ssim_val = calculate_ssim(clean_np[i, 0], enhanced[i, 0])

                results.append({
                    "idx": idx_in_batch,
                    "snr": snr,
                    "psnr": psnr_val,
                    "si_snr": si_snr_val,
                    "mse": mse_val,
                    "correlation": correlation_val,
                    "peak_error": peak_error_val,
                    "ssim": ssim_val
                })

                # 选择样本用于绘图
                sample_data = {
                    "noisy": noisy_np[i, 0],
                    "enhanced": enhanced[i, 0],
                    "clean": clean_np[i, 0],
                    "psnr": psnr_val,
                    "si_snr": si_snr_val,
                    "mse": mse_val,
                    "correlation": correlation_val,
                    "peak_error": peak_error_val,
                    "ssim": ssim_val
                }

                if snr < -5 and len(snr_ranges["Low (-15~-5dB)"]) < 2: # 每个区间取2个样本
                    snr_ranges["Low (-15~-5dB)"].append(sample_data)
                elif -5 <= snr < 5 and len(snr_ranges["Medium (-5~5dB)"]) < 2:
                    snr_ranges["Medium (-5~5dB)"].append(sample_data)
                elif snr >= 5 and len(snr_ranges["High (5~15dB)"]) < 2:
                    snr_ranges["High (5~15dB)"].append(sample_data)
            
            # 显式释放内存
            del noisy, clean, enhanced, noisy_np, clean_np
            gc.collect()

    # --- 打印汇总指标 ---
    all_psnr = [r['psnr'] for r in results]
    all_si_snr = [r['si_snr'] for r in results]
    all_mse = [r['mse'] for r in results]
    all_correlation = [r['correlation'] for r in results]
    all_peak_error = [r['peak_error'] for r in results]
    all_ssim = [r['ssim'] for r in results]
    
    print(f"\n--- 测试汇总 ---")
    print(f"平均 PSNR: {np.mean(all_psnr):.2f} dB")
    print(f"平均 SI-SNR: {np.mean(all_si_snr):.2f} dB")
    print(f"平均 MSE: {np.mean(all_mse):.6f}")
    print(f"平均 互相关系数: {np.mean(all_correlation):.4f}")
    print(f"平均 峰值位置误差: {np.mean(all_peak_error):.2f} 采样点")
    print(f"平均 SSIM: {np.mean(all_ssim):.4f}")

    # --- 绘制波形对比图 ---
    plot_count = 1
    for snr_label, samples in snr_ranges.items():
        for i, sample in enumerate(samples):
            # 创建新图
            plt.figure(figsize=(12, 12))
            t = np.arange(len(sample['noisy']))
            
            # 设置y轴范围
            # 噪声信号使用动态范围
            noisy_min = np.min(sample['noisy']) * 1.1
            noisy_max = np.max(sample['noisy']) * 1.1
            # 干净信号和增强信号使用固定范围[-1.5, 1.5]
            clean_enhanced_min = -1.5
            clean_enhanced_max = 1.5
            
            # 子图1: 噪声信号
            plt.subplot(3, 1, 1)
            plt.plot(t, sample['noisy'], color='red')
            plt.title(f"{snr_label} - 样本 {i+1} | 噪声信号 (Noisy Input)", fontsize=14)
            plt.ylabel('信号幅度', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.ylim(noisy_min, noisy_max)
            
            # 子图2: 干净信号
            plt.subplot(3, 1, 2)
            plt.plot(t, sample['clean'], color='blue')
            plt.title(f"干净信号 (Clean Target)", fontsize=14)
            plt.ylabel('信号幅度', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.ylim(clean_enhanced_min, clean_enhanced_max)
            
            # 子图3: 增强信号
            plt.subplot(3, 1, 3)
            plt.plot(t, sample['enhanced'], color='green', linestyle='--')
            plt.title(f"增强信号 (Enhanced Output) | PSNR: {sample['psnr']:.1f}dB | SI-SNR: {sample['si_snr']:.1f}dB\nMSE: {sample['mse']:.4f} | 互相关: {sample['correlation']:.2f} | 峰值误差: {sample['peak_error']:.1f} | SSIM: {sample['ssim']:.2f}", fontsize=14)
            plt.xlabel('采样点索引', fontsize=12)
            plt.ylabel('信号幅度', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.ylim(clean_enhanced_min, clean_enhanced_max)
            
            plt.tight_layout()
            
            # 保存单独的图片
            snr_folder = snr_label.replace('~', '_').replace('-', '_').replace('(', '').replace(')', '')
            save_path = os.path.join(OUTPUT_DIR, f"enhancement_{snr_folder}_{i+1}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"波形对比图已保存至: {save_path}")
            
            plt.close()  # 关闭当前图，释放内存
            plot_count += 1

if __name__ == "__main__":
    main()