import os
import re
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

def _normalize_to_us_short(ts):
    """与 samples 文件中使用的 normalize 保持一致（简短版）"""
    if ts is None:
        return None
    try:
        t = float(ts)
    except Exception:
        s = re.findall(r"[\d]+\.[\d]+|[\d]+", str(ts))
        if not s:
            return None
        t = float(s[0])
    if t < 1e11:
        return int(round(t * 1e6))
    elif t < 1e14:
        return int(round(t * 1e3))
    else:
        return int(round(t))

def get_sequence_time_range(seq_path):
    """获取 sequence 的时间范围 (us)"""
    lidar_dir = os.path.join(seq_path, "lidar")
    if not os.path.exists(lidar_dir):
        return None, None
    
    files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.bin')])
    if not files:
        return None, None
        
    t_start = _normalize_to_us_short(Path(files[0]).stem)
    t_end = _normalize_to_us_short(Path(files[-1]).stem)
    return t_start, t_end


def preprocess_localization_and_clean(localization_dir, seq_data_root, sweep_root):
    print("\n=== [预处理] 正在扫描 localization.csv 并尝试抢救有效数据 ===")
    
    # 1. 收集所有 localization.csv
    loc_files = []
    if os.path.exists(localization_dir):
        for root, dirs, files in os.walk(localization_dir):
            if "localization.csv" in files:
                loc_files.append(os.path.join(root, "localization.csv"))
    
    if not loc_files:
        print("未找到 localization.csv 文件")
        return

    # 2. 逐个处理
    for csv_path in loc_files:
        # 推断 bev_date
        path_obj = Path(csv_path)
        bev_date = None
        for p in path_obj.parents:
            if p.name.startswith("bev_date_"):
                bev_date = p.name
                break
        
        if not bev_date:
            continue

        print(f"\n正在检查: {bev_date} ({csv_path})")
        
        # --- A. 读取并清洗 CSV ---
        valid_segments = [] # 存储有效的连续时间段 [(start_us, end_us), ...]
        df_clean = None
        is_csv_broken = False
        
        try:
            # 读取，跳过坏行
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            if df.empty:
                print("  -> CSV 为空")
                is_csv_broken = True
            else:
                # 找时间戳列
                cols = [c.lower() for c in df.columns]
                ts_col_idx = -1
                for i, c in enumerate(cols):
                    if 'timestamp' in c:
                        ts_col_idx = i
                        break
                
                if ts_col_idx == -1:
                    print("  -> 缺少时间戳列")
                    is_csv_broken = True
                else:
                    # 强制转数字，清洗非数字行
                    # 注意：这里我们保留原始 df 的副本用于后续保存，但只基于有效时间戳行进行分析
                    # 为了简单，我们假设如果时间戳无效，整行都无效，直接丢弃
                    df['ts_numeric'] = pd.to_numeric(df.iloc[:, ts_col_idx], errors='coerce')
                    df_clean = df.dropna(subset=['ts_numeric']).copy()
                    
                    if df_clean.empty:
                        print("  -> 所有行的时间戳均无效")
                        is_csv_broken = True
                    else:
                        # 排序
                        df_clean = df_clean.sort_values(by='ts_numeric')
                        
                        # 统一转微秒
                        ts_values = df_clean['ts_numeric'].values
                        sample_val = ts_values[0]
                        if sample_val < 1e11: # 秒
                            ts_us = (ts_values * 1e6).astype(np.int64)
                        elif sample_val < 1e14: # 毫秒
                            ts_us = (ts_values * 1e3).astype(np.int64)
                        else: # 微秒
                            ts_us = ts_values.astype(np.int64)
                        
                        # --- B. 切分连续片段 ---
                        GAP_THRESHOLD = 350_000 # 350ms
                        
                        # 计算 diff
                        diffs = np.diff(ts_us)
                        # 找到断点索引 (diff > threshold 的位置)
                        # np.where 返回的是 tuple
                        break_indices = np.where(diffs > GAP_THRESHOLD)[0]
                        
                        # 构建片段
                        start_idx = 0
                        for break_idx in break_indices:
                            # break_idx 是 diff 的索引，对应 ts_us[break_idx] 和 ts_us[break_idx+1] 之间断开
                            # 所以当前片段是 start_idx 到 break_idx (包含)
                            end_idx = break_idx
                            valid_segments.append((ts_us[start_idx], ts_us[end_idx]))
                            start_idx = break_idx + 1
                        
                        # 最后一个片段
                        valid_segments.append((ts_us[start_idx], ts_us[-1]))
                        
                        print(f"  -> 发现 {len(valid_segments)} 个连续定位片段")

        except Exception as e:
            print(f"  -> 读取 CSV 失败: {e}")
            is_csv_broken = True

        # --- C. 扫描 Sequence 并匹配 ---
        # 无论 CSV 是否损坏，都要扫描 sequence，以便决定移动哪些
        
        seq_status = [] # [(seq_name, seq_path, is_safe, reason), ...]
        
        search_roots = []
        # 1. 扫描 seq_data_root (混合结构)
        if seq_data_root and os.path.exists(seq_data_root):
            # 模式1: bev_date/sequence (层级)
            p1 = os.path.join(seq_data_root, bev_date)
            if os.path.exists(p1):
                for item in os.listdir(p1):
                    if item.startswith("sequence"):
                        seq_path = os.path.join(p1, item)
                        seq_status.append(check_sequence(item, seq_path, valid_segments, is_csv_broken))

            # 模式2: bev_date_sequence (扁平)
            # 扫描 seq_data_root 下所有以当前 bev_date 开头的文件
            for item in os.listdir(seq_data_root):
                # 匹配 bev_date_xxx_sequence...
                # 确保 item 是以 bev_date 开头，并且后面紧跟 _sequence
                # 例如 bev_date="bev_date_20251110170624"
                # item="bev_date_20251110170624_sequence00000"
                if item.startswith(bev_date) and "_sequence" in item:
                    # 提取 sequence 名 (例如 sequence00000)
                    # 假设格式固定为 bev_date + "_" + sequence_name
                    expected_prefix = bev_date + "_"
                    if item.startswith(expected_prefix):
                        seq_name = item[len(expected_prefix):] # 提取出 sequence00000
                        seq_path = os.path.join(seq_data_root, item)
                        seq_status.append(check_sequence(seq_name, 
                                    seq_path, valid_segments, is_csv_broken))

        # 去重 (可能同一个 sequence 被扫到多次，或者模式1模式2重叠)
        unique_status = {}
        for s in seq_status:
            unique_status[s[1]] = s # 以 path 为 key
        
        # --- D. 执行操作 ---
        safe_count = 0
        unsafe_seqs = []
        
        for _, (name, path, is_safe, reason) in unique_status.items():
            if is_safe:
                safe_count += 1
            else:
                unsafe_seqs.append((name, path, reason))
        
        print(f"  -> 扫描结果: {safe_count} 个安全, {len(unsafe_seqs)} 个问题 sequence")
        
        # 1. 如果有安全的 sequence，且 CSV 被清洗过 (df_clean 存在)，则保存清洗后的 CSV
        if safe_count > 0 and df_clean is not None:
            # 检查是否需要重写 (行数变少说明有脏数据被剔除)
            # 或者为了保险，总是重写清洗后的版本（去除了空行、乱码行）
            # 注意：df_clean 包含 'ts_numeric' 辅助列，保存前要删掉
            try:
                # 备份原文件
                backup_path = csv_path + ".bak"
                if not os.path.exists(backup_path):
                    shutil.copy2(csv_path, backup_path)
                
                # 保存清洗后的数据
                # 恢复原始列（去掉 ts_numeric）
                cols_to_save = [c for c in df_clean.columns if c != 'ts_numeric']
                df_clean[cols_to_save].to_csv(csv_path, index=False)
                print(f"  -> [修复] 已保存清洗后的 localization.csv (原文件备份为 .bak)")
            except Exception as e:
                print(f"  -> [警告] 保存清洗后的 CSV 失败: {e}")

        # 2. 移动不安全的 sequence
        if unsafe_seqs:
            print(f"  -> 正在移动 {len(unsafe_seqs)} 个问题 sequence 到 deleted 文件夹...")
            
            # 准备 deleted 目录
            # 假设 seq_data_root 的同级目录
            parent_dir = os.path.dirname(seq_data_root.rstrip('/'))
            del_seq_dir = os.path.join(parent_dir, "deleted_seq_data")
            os.makedirs(del_seq_dir, exist_ok=True)
            
            # 如果有 sweep_root，也要准备
            del_sweep_dir = None
            if sweep_root:
                parent_sweep = os.path.dirname(sweep_root.rstrip('/'))
                del_sweep_dir = os.path.join(parent_sweep, "deleted_sweep_data")
                os.makedirs(del_sweep_dir, exist_ok=True)

            for name, path, reason in unsafe_seqs:
                print(f"    [移动] {path}")
                print(f"      原因: {reason}")
                
                # 移动 seq_data
                # 保持原有结构：如果是 bev_date/seq，则在 deleted 下也创建 bev_date/seq
                # 如果是 bev_date_seq，则直接移动
                
                # 计算相对路径以保持结构
                try:
                    rel_path = os.path.relpath(path, seq_data_root)
                except ValueError:
                    rel_path = os.path.basename(path)

                dst_path = os.path.join(del_seq_dir, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                try:
                    if os.path.exists(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.move(path, dst_path)
                except Exception as e:
                    print(f"      移动失败: {e}")

                # 尝试移动对应的 sweep 数据
                if sweep_root:
                    # 构造 sweep 下可能的路径
                    # 假设结构相同
                    sweep_rel_path = os.path.join(bev_date, name)
                    sweep_src = os.path.join(sweep_root, sweep_rel_path)
                    
                    if os.path.exists(sweep_src):
                        # 移动到 deleted_sweep/bev_date/sequence00000
                        sweep_dst = os.path.join(del_sweep_dir, sweep_rel_path)
                        os.makedirs(os.path.dirname(sweep_dst), exist_ok=True)
                        try:
                            if os.path.exists(sweep_dst):
                                shutil.rmtree(sweep_dst)
                            shutil.move(sweep_src, sweep_dst)
                            print(f"      [同步移动 Sweep] {sweep_src}")
                        except Exception as e:
                            print(f"      移动 Sweep 失败: {e}")
                    else:
                        # 尝试扁平路径 (万一 sweep 也是扁平的)
                        flat_name = f"{bev_date}_{name}"
                        sweep_src_flat = os.path.join(sweep_root, flat_name)
                        if os.path.exists(sweep_src_flat):
                            sweep_dst_flat = os.path.join(del_sweep_dir, flat_name)
                            os.makedirs(os.path.dirname(sweep_dst_flat), exist_ok=True)
                            try:
                                if os.path.exists(sweep_dst_flat):
                                    shutil.rmtree(sweep_dst_flat)
                                shutil.move(sweep_src_flat, sweep_dst_flat)
                                print(f"      [同步移动 Sweep] {sweep_src_flat}")
                            except Exception as e:
                                print(f"      移动 Sweep 失败: {e}")

def check_sequence(seq_name, seq_path, valid_segments, is_csv_broken):
    """
    检查单个 sequence 是否有效
    返回: (seq_name, seq_path, is_safe, reason)
    """
    if is_csv_broken:
        return seq_name, seq_path, False, "Localization CSV 损坏或无效"
    
    t_start, t_end = get_sequence_time_range(seq_path)
    
    if t_start is None:
        return seq_name, seq_path, False, "无法获取 LiDAR 时间范围 (空文件夹?)"
    
    # 检查是否完全落在一个连续片段内
    # 允许 0.08s 的边缘容忍度 (与 ego_pose 逻辑一致)
    TOLERANCE = 80_000 
    
    is_covered = False
    for seg_start, seg_end in valid_segments:
        # 检查包含关系
        # seg_start - tol <= t_start  AND  t_end <= seg_end + tol
        if (seg_start - TOLERANCE <= t_start) and (t_end <= seg_end + TOLERANCE):
            is_covered = True
            break
    
    if is_covered:
        return seq_name, seq_path, True, "OK"
    else:
        # 详细原因
        reason = f"时间范围 [{t_start}, {t_end}] 未被任何连续定位片段覆盖"
        return seq_name, seq_path, False, reason

if __name__ == "__main__":
    # 示例用法
    localization_dir = "/home/pix/prog_doc/job_data/bev_data"
    seq_data_root = "/media/pix/Elements SE/Datasets/Nuscenes/mini/seq_data"
    sweep_root = "/home/pix/prog_doc/job_data/bev_data"
    
    preprocess_localization_and_clean(localization_dir, seq_data_root, sweep_root)
