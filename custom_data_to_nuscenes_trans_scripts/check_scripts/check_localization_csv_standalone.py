import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# === 核心算法 ===
def calculate_metrics(x, y, dt_sec=0.01):
    # 1. 微分计算
    dx = np.diff(x)
    dy = np.diff(y)
    ddx = np.diff(dx)
    ddy = np.diff(dy)
    # 加速度 (未平滑)
    acc_raw = np.sqrt(ddx**2 + ddy**2) / (dt_sec ** 2)
    
    # 2. 噪声强度 (RMSE)
    # 使用小窗口(5)平滑作为基准线，计算原始数据偏离基准线的程度
    window = 5
    if len(x) > window:
        x_smooth = pd.Series(x).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        y_smooth = pd.Series(y).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values
        
        noise_x = x - x_smooth
        noise_y = y - y_smooth
        noise_rmse = np.sqrt(np.mean(noise_x**2) + np.mean(noise_y**2))
    else:
        noise_rmse = 0.0

    # 3. 锯齿率 (ZigZag) - 检测 INS 是否失稳震荡
    # 统计 x 方向差分的符号变化次数
    valid_diff_x = dx[np.abs(dx) > 1e-4]
    if len(valid_diff_x) > 1:
        sign_changes_x = np.sum(np.diff(np.sign(valid_diff_x)) != 0)
        zigzag_rate = sign_changes_x / len(valid_diff_x)
    else:
        zigzag_rate = 0.0

    return {
        "max_acc": np.max(acc_raw) if len(acc_raw) > 0 else 0,
        "mean_acc": np.mean(acc_raw) if len(acc_raw) > 0 else 0,
        "rmse": noise_rmse,
        "zigzag": zigzag_rate
    }

def check_csv(file_path, output_dir=None):
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
    except Exception as e:
        return {"file": filename, "status": "ERROR", "action": "Manual Check", "msg": f"Read Fail: {e}"}

    # === 列名与时间戳处理 ===
    cols = df.columns
    time_col = None
    time_unit = None
    
    if 'timestamp_us' in cols: time_col, time_unit = 'timestamp_us', 'us'
    elif 'timestamp_ms' in cols: time_col, time_unit = 'timestamp_ms', 'ms'
    elif 'timestamp_s' in cols: time_col, time_unit = 'timestamp_s', 's'
    else:
        for c in cols:
            if 'timestamp' in c: time_col = c; break

    if time_col is None:
        return {"file": filename, "status": "ERROR", "action": "Discard", "msg": "No timestamp col"}

    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col]).sort_values(by=time_col).reset_index(drop=True)
    
    ts = df[time_col].values.astype(float)
    if len(ts) < 10:
        return {"file": filename, "status": "ERROR", "action": "Discard", "msg": "Data too short"}

    # 时间单位归一化到秒
    if time_unit == 's': t_sec = ts
    elif time_unit == 'ms': t_sec = ts / 1e3
    elif time_unit == 'us': t_sec = ts / 1e6
    else:
        # Fallback (基于数值大小的安全判断)
        if ts[0] > 1e14: t_sec = ts / 1e6
        elif ts[0] > 1e11: t_sec = ts / 1e3
        else: t_sec = ts
    
    # === 坐标投影 ===
    if 'lat' in cols and 'lon' in cols:
        lat = pd.to_numeric(df['lat'], errors='coerce').values
        lon = pd.to_numeric(df['lon'], errors='coerce').values
        RE = 6378137.0
        x = np.radians(lon - lon[0]) * RE * np.cos(np.radians(lat[0]))
        y = np.radians(lat - lat[0]) * RE
    elif 'x' in cols and 'y' in cols:
        x = pd.to_numeric(df['x'], errors='coerce').values
        y = pd.to_numeric(df['y'], errors='coerce').values
        x = x - x[0]
        y = y - y[0]
    else:
        return {"file": filename, "status": "ERROR", "action": "Discard", "msg": "No lat/lon or x/y"}

    # 计算 dt
    dt_arr = np.diff(t_sec)
    dt_mean = np.mean(dt_arr) if len(dt_arr) > 0 else 0.01
    dt_mean = max(0.001, dt_mean)

    # === 计算指标 ===
    metrics = calculate_metrics(x, y, dt_mean)
    
    # === 【关键：智能分级逻辑】 ===
    status = ""
    action = ""
    msg = ""
    
    # 阈值定义 (单位: 米)
    RMSE_PERFECT = 0.03   # 3cm以内: 完美
    RMSE_FIXABLE = 0.15   # 15cm以内: 可通过平滑修复
    
    # 逻辑树
    if metrics['rmse'] < RMSE_PERFECT:
        # 情况A: 极高精度 (即便加速度很大，也是微分噪声)
        status = "[OK]"
        action = "Ready"
        msg = f"Precise(Noise={metrics['rmse']*100:.1f}cm)"
        if metrics['max_acc'] > 500:
             msg += " w/ Spikes" # 只是有个别飞点，也不怕

    elif metrics['rmse'] < RMSE_FIXABLE:
        # 情况B: 精度一般，或有噪声/跳变
        if metrics['zigzag'] > 0.4:
            # 持续震荡 -> 可能是 INS 解算发散
            status = "[FATAL]"
            action = "Discard"
            msg = f"Unstable Oscillation"
        else:
            # 只是噪声大或有个别大跳变 -> 软件能修
            status = "[FIXABLE]"
            action = "Auto-Smooth"
            msg = f"Noisy({metrics['rmse']*100:.1f}cm)"
            
    else:
        # 情况C: 误差过大 (>15cm)
        status = "[FATAL]"
        action = "Discard"
        msg = f"Large Drift({metrics['rmse']*100:.0f}cm)"

    # === 绘图 (仅针对 Warning/Fatal 生成图片以便人工复核) ===
    if output_dir and "OK" not in status:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.plot(x, y, label='Traj')
        plt.title(f"{filename[:20]}...\n{status} - {msg}")
        plt.axis('equal')
        plt.subplot(122)
        # 画加速度谱
        acc_plot = np.diff(np.diff(x)) / dt_mean**2
        plt.plot(t_sec[:-2]-t_sec[0], acc_plot, label='Acc X', alpha=0.7)
        plt.title(f"Acc Noise (Max={metrics['max_acc']:.0f})")
        plt.ylim(-200, 200) # 限制视野以免飞点压缩图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Check_{filename}.png"))
        plt.close()

    return {
        "file": filename,
        "status": status,
        "action": action,
        "rmse_cm": metrics['rmse'] * 100,
        "max_acc": metrics['max_acc'],
        "zigzag": metrics['zigzag'],
        "msg": msg
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str, nargs='?', 
                        default='.', help='Input file or directory')
    args = parser.parse_args()
    
    files_to_check = []
    if os.path.isfile(args.csv_path):
        files_to_check.append(args.csv_path)
    elif os.path.isdir(args.csv_path):
        for root, _, files in os.walk(args.csv_path):
            files.sort()
            for f in files:
                if f.endswith('.csv') and 'localization' in f:
                    files_to_check.append(os.path.join(root, f))
                elif f.endswith('.csv') and 'perception_data' in f:
                    files_to_check.append(os.path.join(root, f))
    
    if not files_to_check:
        print("No CSV files found.")
        return

    results = []
    print(f"Checking {len(files_to_check)} files...")
    
    for f in tqdm(files_to_check, disable=len(files_to_check)<5):
        res = check_csv(f, output_dir='./check_localization_bad_plots')
        results.append(res)
    
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print(f"DATA HEALTH ASSESSMENT REPORT (Total: {len(results)})")
    print("="*100)
    
    # 精简文件名显示
    df_res['file_disp'] = df_res['file'].apply(lambda x: x[-30:] if len(x)>3