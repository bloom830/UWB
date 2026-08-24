"""
batch_test.py - UWB 多信噪比批量测试脚本
功能：
  1. 在不同 SNR (-15/-12/-9/-6/-3/0/3/6/9/12/15 dB) 测试集上批量推理
  2. 计算每个 SNR 下的 MSE / RMSE / 峰值误差 / SSIM 等指标
  3. 保存去噪后的信号 (.npy / .pt) 到 batch_test_results 目录
  4. 生成波形对比图 (clean vs noisy vs denoised)
数据特征：
  - 每个 SNR 对应 data/uwb_comprehensive_test_set/snr_xxxdb/ 下
  - 单通道实数，长度 4048
  - 输出每个 SNR 的综合指标表供分析脚本进一步汇总
典型使用：
  python testing/batch_test.py
"""
import sys, os
# 确保项目根目录加入 sys.path，支持从任意目录运行本脚本
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import yaml
from data_loader.test_dataload import TestDataset, create_test_loader
from models.UWB_DeFT_AN import UWB_Network
from skimage.metrics import structural_similarity as ssim

# 解决OpenMP重复初始化问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 指标计算函数 --- 
def calculate_snr(target, prediction, eps=1e-8):
    """计算标准信噪比 SNR = 10 * log10(信号功率 / 噪声功率)"""
    signal_power = np.mean(target ** 2)
    noise = prediction - target
    noise_power = np.mean(noise ** 2)
    if noise_power < eps:
        return float('inf')
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_snr_improvement(input_snr_db, output_snr_db):
    """计算SNR提升量"""
    return output_snr_db - input_snr_db

def calculate_mse(target, prediction):
    """计算均方误差"""
    return np.mean((target - prediction) ** 2)

def calculate_correlation(target, prediction):
    """计算互相关系数"""
    target_norm = target - np.mean(target)
    prediction_norm = prediction - np.mean(prediction)
    correlation = np.dot(target_norm, prediction_norm) / (np.linalg.norm(target_norm) * np.linalg.norm(prediction_norm))
    return correlation

def calculate_peak_error(target, prediction, threshold_ratio=0.1, avg_pulse_spacing=None):
    """计算所有显著峰值的平均位置误差、正负峰值召回率和峰值误差比
    
    Args:
        target: 参考信号
        prediction: 增强信号
        threshold_ratio: 显著峰值阈值比例（相对于最大峰值的比值）
        avg_pulse_spacing: 平均脉冲间距（用于归一化误差比，None则用估算值）
    
    Returns:
        (平均峰值位置误差, 峰值误差比(0~1), 正峰值召回率, 负峰值召回率, 总召回率)
    """
    from scipy.signal import find_peaks
    
    max_val = np.max(np.abs(target))
    if max_val < 1e-10:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    threshold = max_val * threshold_ratio
    
    # 分别检测正峰值和负峰值
    target_pos_peaks, _ = find_peaks(target, height=threshold, distance=10)
    target_neg_peaks, _ = find_peaks(-target, height=threshold, distance=10)
    pred_pos_peaks, _ = find_peaks(prediction, height=threshold, distance=10)
    pred_neg_peaks, _ = find_peaks(-prediction, height=threshold, distance=10)
    
    # 估算平均脉冲间距：信号长度 / 总脉冲数
    total_pulses = len(target_pos_peaks) + len(target_neg_peaks)
    if avg_pulse_spacing is None:
        if total_pulses > 0:
            avg_pulse_spacing = len(target) / total_pulses
        else:
            avg_pulse_spacing = 1.0
    
    def match_peaks(target_peaks, pred_peaks, target_signal, pred_signal):
        """匹配峰值，要求极性一致"""
        if len(target_peaks) == 0:
            return 0, 0
        if len(pred_peaks) == 0:
            return 0, len(target_peaks)
        
        errors = []
        used_pred = set()
        
        for t_peak in target_peaks:
            min_dist = float('inf')
            best_p_peak = None
            
            # 获取参考信号在该峰值处的极性
            target_polarity = np.sign(target_signal[t_peak])
            
            for p_idx, p_peak in enumerate(pred_peaks):
                if p_idx in used_pred:
                    continue
                
                # 检查预测信号在该峰值处的极性是否与参考信号一致
                pred_polarity = np.sign(pred_signal[p_peak])
                if target_polarity * pred_polarity <= 0:  # 极性不一致或为零
                    continue
                
                dist = abs(t_peak - p_peak)
                if dist < min_dist:
                    min_dist = dist
                    best_p_peak = p_idx
            
            if best_p_peak is not None:
                errors.append(min_dist)
                used_pred.add(best_p_peak)
        
        return len(errors), len(target_peaks)
    
    pos_matched, pos_total = match_peaks(target_pos_peaks, pred_pos_peaks, target, prediction)
    neg_matched, neg_total = match_peaks(target_neg_peaks, pred_neg_peaks, target, prediction)
    
    pos_recall = pos_matched / pos_total if pos_total > 0 else 0.0
    neg_recall = neg_matched / neg_total if neg_total > 0 else 0.0
    
    total_matched = pos_matched + neg_matched
    total_target = pos_total + neg_total
    comprehensive_recall = total_matched / total_target if total_target > 0 else 0.0
    
    # 计算平均位置误差（也需要极性一致）
    all_errors = []
    used_pred_all = set()
    
    for t_peak in np.concatenate([target_pos_peaks, target_neg_peaks]):
        min_dist = float('inf')
        best_p_peak = None
        all_pred_peaks = np.concatenate([pred_pos_peaks, pred_neg_peaks])
        
        target_polarity = np.sign(target[t_peak])
        
        for p_idx, p_peak in enumerate(all_pred_peaks):
            if p_idx in used_pred_all:
                continue
            
            pred_polarity = np.sign(prediction[p_peak])
            if target_polarity * pred_polarity <= 0:
                continue
            
            dist = abs(t_peak - p_peak)
            if dist < min_dist:
                min_dist = dist
                best_p_peak = p_idx
        
        if best_p_peak is not None:
            all_errors.append(min_dist)
            used_pred_all.add(best_p_peak)
    
    avg_error = np.mean(all_errors) if len(all_errors) > 0 else 0.0
    peak_error_ratio = min(avg_error / avg_pulse_spacing, 1.0) if avg_pulse_spacing > 0 else 1.0
    
    return avg_error, peak_error_ratio, pos_recall, neg_recall, comprehensive_recall

def calculate_ssim(target, prediction):
    """计算结构相似性"""
    try:
        s = ssim(target, prediction, data_range=target.max()-target.min())
        return s
    except:
        return 0.0

# --- 2. 批量测试主函数 --- 
def main():
    # --- 加载配置文件 --- 
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_script_dir, "test.yaml"), 'r', encoding='utf-8') as f:
        test_config = yaml.safe_load(f)
    
    # --- 配置参数 --- 
    DATA_ROOT = test_config['DATA_ROOT']
    MODEL_PATH = test_config['MODEL_PATH']
    OUTPUT_DIR = test_config['OUTPUT_DIR']
    ENHANCED_DIR = test_config['ENHANCED_DIR']
    RESULT_FILE = test_config['RESULT_FILE']
    BATCH_SIZE = test_config['BATCH_SIZE']
    TARGET_SNRS = test_config['TARGET_SNRS']
    
    # 模型架构参数
    MODEL_PARAMS = test_config['MODEL']
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 加载测试集 --- 
    print("="*60)
    print("加载综合SNR测试集 (-15dB 到 +15dB，步长3dB)...")
    
    # 使用新的测试数据加载器
    test_loader = create_test_loader(DATA_ROOT, BATCH_SIZE, TARGET_SNRS)
    
    # 获取测试集对象以访问SNR信息
    test_dataset = test_loader.dataset
    
    # 加载SNR信息
    snr_info_path = os.path.join(DATA_ROOT, "snr_info.json")
    with open(snr_info_path, 'r') as f:
        snr_info = json.load(f)
    test_snr_list = snr_info['test_snr']
    
    # 获取所有唯一的SNR值
    unique_snrs = sorted(list(set(test_snr_list)))
    print(f"测试集包含的SNR值: {unique_snrs} dB")
    print(f"总样本数: {len(test_dataset)}")
    print("="*60)
    
    # --- 初始化模型 --- 
    # 使用配置文件中的模型架构参数
    model = UWB_Network(
        win=MODEL_PARAMS['win'],
        hop=MODEL_PARAMS['hop'],
        ch_dim=MODEL_PARAMS['ch_dim'],
        num_layer=MODEL_PARAMS['num_layer'],
        depth=MODEL_PARAMS['depth']
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"模型加载完毕: {MODEL_PATH}")
    
    # --- 推理与指标计算 --- 
    print("\n开始批量测试各SNR性能...")
    
    # 按SNR分组存储结果和增强数据
    snr_results = {snr: [] for snr in unique_snrs}
    enhanced_data = {snr: [] for snr in unique_snrs}  # 存储增强后的数据
    
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
                
                # 获取当前样本的SNR
                snr = test_snr_list[idx_in_batch]
                
                # 计算所有指标（增强后）
                snr_val = calculate_snr(clean_np[i, 0], enhanced[i, 0])
                snr_improve_val = calculate_snr_improvement(snr, snr_val)
                mse_val = calculate_mse(clean_np[i, 0], enhanced[i, 0])
                correlation_val = calculate_correlation(clean_np[i, 0], enhanced[i, 0])
                peak_error_val, peak_error_ratio_val, pos_recall_val, neg_recall_val, comprehensive_recall_val = calculate_peak_error(clean_np[i, 0], enhanced[i, 0])
                ssim_val = calculate_ssim(clean_np[i, 0], enhanced[i, 0])
                
                # 计算增强前指标（noisy vs clean）
                corr_before_val = calculate_correlation(clean_np[i, 0], noisy_np[i, 0])
                peak_err_before_val, peak_err_ratio_before_val, pos_recall_before_val, neg_recall_before_val, recall_before_val = calculate_peak_error(clean_np[i, 0], noisy_np[i, 0])
                ssim_before_val = calculate_ssim(clean_np[i, 0], noisy_np[i, 0])

                # 按SNR分组存储结果
                snr_results[snr].append({
                    "snr": snr_val,
                    "snr_improve": snr_improve_val,
                    "mse": mse_val,
                    "correlation": correlation_val,
                    "correlation_before": corr_before_val,
                    "peak_error": peak_error_val,
                    "peak_error_before": peak_err_before_val,
                    "peak_error_ratio": peak_error_ratio_val,
                    "peak_error_ratio_before": peak_err_ratio_before_val,
                    "pos_peak_recall": pos_recall_val,
                    "pos_peak_recall_before": pos_recall_before_val,
                    "neg_peak_recall": neg_recall_val,
                    "neg_peak_recall_before": neg_recall_before_val,
                    "comprehensive_recall": comprehensive_recall_val,
                    "comprehensive_recall_before": recall_before_val,
                    "ssim": ssim_val,
                    "ssim_before": ssim_before_val
                })
                
                # 保存增强后的数据
                enhanced_data[snr].append(enhanced[i, 0])
    
    # --- 结果汇总与保存 --- 
    print("\n" + "="*60)
    print("各SNR性能汇总:")
    print("="*60)
    
    # 保存结果到txt文件
    result_file = os.path.join(OUTPUT_DIR, RESULT_FILE)
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("UWB信号增强模型批量测试结果 (综合测试集 - 优化后模型)\n")
        f.write("模型: train_log5(416) | 架构: win=128, hop=32, ch_dim=48, num_layer=6, depth=4\n")
        f.write("信噪比范围: -15dB 到 +15dB，步长3dB\n")
        f.write("峰值误差: 所有显著峰值的平均位置误差（阈值=最大峰值10%）\n")
        f.write("峰值误差比: 平均位置误差 / 平均脉冲间距 (0~1，越小越好)\n")
        f.write("="*145 + "\n\n")
        
        # 表头（与 0714modeltest.txt 完全一致）
        f.write("SNR_in(dB) | 样本数 | SNR_out(dB) | SNR提升(dB) | MSE | 互相关(后) | 互相关(前) | 峰值误差(后) | 峰值误差(前) | 峰值误差比(后) | 峰值误差比(前) | 正峰召回(后) | 正峰召回(前) | 负峰召回(后) | 负峰召回(前) | 综合召回(后) | 综合召回(前) | SSIM(后) | SSIM(前)\n")
        f.write("-"*240 + "\n")
        
        # 计算每个SNR的平均性能
        for snr in unique_snrs:
            results = snr_results[snr]
            num_samples = len(results)
            
            avg_snr = np.mean([r['snr'] for r in results])
            avg_snr_improve = np.mean([r['snr_improve'] for r in results])
            avg_mse = np.mean([r['mse'] for r in results])
            avg_correlation = np.mean([r['correlation'] for r in results])
            avg_corr_before = np.mean([r['correlation_before'] for r in results])
            avg_peak_error = np.mean([r['peak_error'] for r in results])
            avg_peak_error_before = np.mean([r['peak_error_before'] for r in results])
            avg_peak_error_ratio = np.mean([r['peak_error_ratio'] for r in results])
            avg_peak_error_ratio_before = np.mean([r['peak_error_ratio_before'] for r in results])
            avg_pos_recall = np.mean([r['pos_peak_recall'] for r in results])
            avg_pos_recall_before = np.mean([r['pos_peak_recall_before'] for r in results])
            avg_neg_recall = np.mean([r['neg_peak_recall'] for r in results])
            avg_neg_recall_before = np.mean([r['neg_peak_recall_before'] for r in results])
            avg_comprehensive_recall = np.mean([r['comprehensive_recall'] for r in results])
            avg_comprehensive_recall_before = np.mean([r['comprehensive_recall_before'] for r in results])
            avg_ssim = np.mean([r['ssim'] for r in results])
            avg_ssim_before = np.mean([r['ssim_before'] for r in results])
            
            # 写入txt文件（19列，与0714版完全一致的手写format行宽）
            f.write(f"{snr:10.1f} | {num_samples:6d} | {avg_snr:11.2f} | {avg_snr_improve:11.2f} | {avg_mse:9.6f} | {avg_correlation:10.4f} | {avg_corr_before:10.4f} | {avg_peak_error:12.2f} | {avg_peak_error_before:12.2f} | {avg_peak_error_ratio:13.4f} | {avg_peak_error_ratio_before:13.4f} | {avg_pos_recall:11.4f} | {avg_pos_recall_before:11.4f} | {avg_neg_recall:11.4f} | {avg_neg_recall_before:11.4f} | {avg_comprehensive_recall:11.4f} | {avg_comprehensive_recall_before:11.4f} | {avg_ssim:7.4f} | {avg_ssim_before:7.4f}\n")
        
        # 计算整体平均性能
        f.write("\n" + "="*150 + "\n")
        f.write("整体平均性能:\n")
        
        all_snr = []
        all_snr_improve = []
        all_mse = []
        all_correlation = []
        all_correlation_before = []
        all_peak_error = []
        all_peak_error_before = []
        all_peak_error_ratio = []
        all_peak_error_ratio_before = []
        all_pos_recall = []
        all_pos_recall_before = []
        all_neg_recall = []
        all_neg_recall_before = []
        all_comprehensive_recall = []
        all_comprehensive_recall_before = []
        all_ssim = []
        all_ssim_before = []
        
        for snr in unique_snrs:
            results = snr_results[snr]
            all_snr.extend([r['snr'] for r in results])
            all_snr_improve.extend([r['snr_improve'] for r in results])
            all_mse.extend([r['mse'] for r in results])
            all_correlation.extend([r['correlation'] for r in results])
            all_correlation_before.extend([r['correlation_before'] for r in results])
            all_peak_error.extend([r['peak_error'] for r in results])
            all_peak_error_before.extend([r['peak_error_before'] for r in results])
            all_peak_error_ratio.extend([r['peak_error_ratio'] for r in results])
            all_peak_error_ratio_before.extend([r['peak_error_ratio_before'] for r in results])
            all_pos_recall.extend([r['pos_peak_recall'] for r in results])
            all_pos_recall_before.extend([r['pos_peak_recall_before'] for r in results])
            all_neg_recall.extend([r['neg_peak_recall'] for r in results])
            all_neg_recall_before.extend([r['neg_peak_recall_before'] for r in results])
            all_comprehensive_recall.extend([r['comprehensive_recall'] for r in results])
            all_comprehensive_recall_before.extend([r['comprehensive_recall_before'] for r in results])
            all_ssim.extend([r['ssim'] for r in results])
            all_ssim_before.extend([r['ssim_before'] for r in results])
        
        total_avg_snr = np.mean(all_snr)
        total_avg_snr_improve = np.mean(all_snr_improve)
        total_avg_mse = np.mean(all_mse)
        total_avg_correlation = np.mean(all_correlation)
        total_avg_correlation_before = np.mean(all_correlation_before)
        total_avg_peak_error = np.mean(all_peak_error)
        total_avg_peak_error_before = np.mean(all_peak_error_before)
        total_avg_peak_error_ratio = np.mean(all_peak_error_ratio)
        total_avg_peak_error_ratio_before = np.mean(all_peak_error_ratio_before)
        total_avg_pos_recall = np.mean(all_pos_recall)
        total_avg_pos_recall_before = np.mean(all_pos_recall_before)
        total_avg_neg_recall = np.mean(all_neg_recall)
        total_avg_neg_recall_before = np.mean(all_neg_recall_before)
        total_avg_comprehensive_recall = np.mean(all_comprehensive_recall)
        total_avg_comprehensive_recall_before = np.mean(all_comprehensive_recall_before)
        total_avg_ssim = np.mean(all_ssim)
        total_avg_ssim_before = np.mean(all_ssim_before)
        
        f.write(f"平均输出SNR: {total_avg_snr:.2f} dB\n")
        f.write(f"平均SNR提升: {total_avg_snr_improve:.2f} dB\n")
        f.write(f"平均 MSE: {total_avg_mse:.6f}\n")
        f.write(f"平均 互相关系数(后): {total_avg_correlation:.4f}\n")
        f.write(f"平均 互相关系数(前): {total_avg_correlation_before:.4f}\n")
        f.write(f"平均 峰值位置误差(后): {total_avg_peak_error:.2f} 采样点\n")
        f.write(f"平均 峰值位置误差(前): {total_avg_peak_error_before:.2f} 采样点\n")
        f.write(f"平均 峰值误差比(后): {total_avg_peak_error_ratio:.4f}\n")
        f.write(f"平均 峰值误差比(前): {total_avg_peak_error_ratio_before:.4f}\n")
        f.write(f"平均 正峰值召回率(后): {total_avg_pos_recall:.4f}\n")
        f.write(f"平均 正峰值召回率(前): {total_avg_pos_recall_before:.4f}\n")
        f.write(f"平均 负峰值召回率(后): {total_avg_neg_recall:.4f}\n")
        f.write(f"平均 负峰值召回率(前): {total_avg_neg_recall_before:.4f}\n")
        f.write(f"平均 综合召回率(后): {total_avg_comprehensive_recall:.4f}\n")
        f.write(f"平均 综合召回率(前): {total_avg_comprehensive_recall_before:.4f}\n")
        f.write(f"平均 SSIM(后): {total_avg_ssim:.4f}\n")
        f.write(f"平均 SSIM(前): {total_avg_ssim_before:.4f}\n")
    
    # --- 保存增强后的数据 --- 
    enhanced_dir = os.path.join(OUTPUT_DIR, ENHANCED_DIR)
    os.makedirs(enhanced_dir, exist_ok=True)
    
    for snr in unique_snrs:
        # 将增强数据转换为numpy数组
        enhanced_array = np.array(enhanced_data[snr])
        
        # 保存为numpy文件
        enhanced_file = os.path.join(enhanced_dir, f"enhanced_snr_{snr}db.npy")
        np.save(enhanced_file, enhanced_array)
        
        # 同时保存为torch张量文件
        enhanced_tensor = torch.from_numpy(enhanced_array)
        enhanced_pt_file = os.path.join(enhanced_dir, f"enhanced_snr_{snr}db.pt")
        torch.save(enhanced_tensor, enhanced_pt_file)
    
    print(f"批量测试完成！结果已保存至: {result_file}")

if __name__ == "__main__":
    main()
