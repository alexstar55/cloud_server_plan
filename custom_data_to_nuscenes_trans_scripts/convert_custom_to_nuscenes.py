
import json
import os,uuid,shutil
import re
import argparse
from collections import defaultdict
import pandas as pd
import numpy as np
from pathlib import Path
from convert_to_nuscenes_main import (
    load_custom_annotations,
    create_nuscenes_categories,
    create_custom_to_nuscenes_mapping,
    create_nuscenes_attributes,
    create_nuscenes_visibility,
    create_nuscenes_sensors,
    create_nuscenes_calibrated_sensors
)
from convert_to_nuscenes_samples_optimized import (
    create_nuscenes_scene,
    create_nuscenes_log,
    create_nuscenes_samples,
    create_nuscenes_instances,
    _extract_bev_seq_from_path,
    create_nuscenes_sample_data_optimized,
    create_nuscenes_ego_poses,
    create_nuscenes_sample_annotations,
    create_nuscenes_map
)

def apply_smoothing(df, window_size=5):
    """
    对位置数据进行滑动窗口平滑处理
    :param window_size: 窗口大小，建议 5-10
    """
    # 确保数据按时间排序
    df = df.sort_values('ts_numeric').reset_index(drop=True)
    # 需要平滑的列
    cols_to_smooth = ['lat', 'lon', 'alt']
    # 检查列是否存在
    valid_cols = [c for c in cols_to_smooth if c in df.columns]
    # 使用 rolling mean 进行平滑
    # min_periods=1 保证开头结尾的数据不丢失
    # center=True 保证不会产生相位滞后 (非常重要!)
    for col in valid_cols:
        df[col] = df[col].rolling(window=window_size, min_periods=1, center=True).mean()
    return df

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
                        # 平滑滤波
                        # 窗口大小建议：如果是100Hz数据，window=10(0.1s)；如果是10Hz，window=3
                        print(f"  -> 应用平滑滤波 (Window=11)...")
                        # original_len = len(df_clean)
                        # 如果采样率是 100Hz，这只有 0.1秒，对于 33g 的平均噪声可能不够。
                        # 故尝试增大到 window_size=20
                        df_clean = apply_smoothing(df_clean, window_size=20)
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
                        # 为了安全，最好先手动备份一下源文件
                        output_csv_path = csv_path 
                        df_clean.to_csv(output_csv_path, index=False)
                        print(f"  -> 已保存平滑后的数据: {output_csv_path}")

        except Exception as e:
            print(f"  -> 读取 CSV 失败: {e}")
            is_csv_broken = True

        # --- C. 扫描 Sequence 并匹配 ---
        # 无论 CSV 是否损坏，都要扫描 sequence，以便决定移动哪些
        
        seq_status = [] # [(seq_name, seq_path, is_safe, reason), ...]
        
        # search_roots = []
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

def filter_annotations_by_existing_sequences(custom_annotations, seq_data_root):
    """
    根据 seq_data_root 中实际存在的 sequence，过滤标注数据。
    用于剔除因 localization 问题被移走的 sequence 的标注。
    """
    print("正在根据实际存在的 sequence 数据过滤标注...")
    
    if not os.path.exists(seq_data_root):
        print(f"警告: seq_data_root 不存在: {seq_data_root}，跳过过滤")
        return custom_annotations

    # 1. 建立现有 sequence 的白名单集合 {(bev_date, sequence_name)}
    existing_seqs = set()
    
    # 扫描 seq_data_root
    for item in os.listdir(seq_data_root):
        item_path = os.path.join(seq_data_root, item)
        if not os.path.isdir(item_path):
            continue
            
        # 情况1: bev_date_xxx (层级结构)
        if item.startswith("bev_date_") and "sequence" not in item: # 避免误判扁平结构
            bev_date = item
            # 扫描子目录
            for sub in os.listdir(item_path):
                if sub.startswith("sequence") and os.path.isdir(os.path.join(item_path, sub)):
                    existing_seqs.add((bev_date, sub))
        
        # 情况2: bev_date_xxx_sequence_xxx (扁平结构)
        # 或者 sequence_xxx (如果没有 bev_date 前缀)
        else:
            # 尝试解析 bev_date_xxx_sequence_xxx
            # 假设格式是 bev_date_YYYYMMDDHHMMSS_sequenceXXXXX
            if "bev_date_" in item and "_sequence" in item:
                # 找到最后一个 "_sequence" 的位置进行分割
                split_idx = item.rfind("_sequence")
                if split_idx != -1:
                    bev = item[:split_idx] # bev_date_2025...
                    seq = item[split_idx+1:] # sequence000...
                    existing_seqs.add((bev, seq))
            
            elif item.startswith("sequence"):
                # 只有 sequence 名的情况 (旧数据?)
                existing_seqs.add((None, item))

    print(f"  -> 发现 {len(existing_seqs)} 个有效 sequence 文件夹")

    # 2. 过滤标注
    filtered_annotations = []
    removed_count = 0
    
    for frame in custom_annotations:
        info_url = frame.get("info", "")
        if not info_url:
            # 没有 info 的帧通常无效，丢弃
            removed_count += 1
            continue
            
        # 从 info_url 提取 (bev_date, seq)
        # info_url 示例: .../bev_date_xxx/sequence000/lidar/xxx.bin
        path_obj = Path(info_url)
        bev, seq = _extract_bev_seq_from_path(path_obj)
        
        is_valid = False
        
        if bev and seq:
            # 1. 标准精确匹配
            if (bev, seq) in existing_seqs:
                is_valid = True
            
            # 2. 尝试模糊匹配 (应对 bev_date 解析差异)
            # 例如: existing_seqs 里是 ('bev_date_2025...', 'sequence000')
            # 但 info_url 解析出来可能是 ('2025...', 'sequence000')
            if not is_valid:
                for ex_bev, ex_seq in existing_seqs:
                    # 只有当 sequence 名完全一致时才尝试匹配 bev
                    if ex_seq == seq:
                        # 检查 bev 是否包含关系 (双向)
                        if ex_bev and (bev in ex_bev or ex_bev in bev):
                            is_valid = True
                            break
            
        elif seq:
            # 只有 seq (旧数据兼容)
            if (None, seq) in existing_seqs:
                is_valid = True
            # 或者是只要 seq 名匹配就行 (放宽条件，防止 bev_date 解析差异)
            else:
                # 检查是否有任意一个 bev_date 包含此 seq
                for ex_bev, ex_seq in existing_seqs:
                    if ex_seq == seq:
                        is_valid = True
                        break

        if is_valid:
            filtered_annotations.append(frame)
        else:
            removed_count += 1

    print(f"  -> 过滤完成: 保留 {len(filtered_annotations)} 帧, 移除 {removed_count} 帧 (因源数据缺失)")
    return filtered_annotations
    
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
def validate_data_consistency(samples, sample_data, sample_annotations, instances):
    """验证数据一致性 - 调整版本"""
    print("=== 数据一致性检查 ===")
    
    # 检查样本数据完整性
    sample_tokens = set(sample["token"] for sample in samples)
    no_data_cnt, no_annotation_cnt = 0, 0
    for sample in samples:
        sample_token = sample["token"]
        sample_data_count = len([sd for sd in sample_data if sd["sample_token"] == sample_token])
        sample_annotation_count = len([sa for sa in sample_annotations if sa["sample_token"] == sample_token])
        
        print(f"样本 {sample_token}: {sample_data_count} 个样本数据, {sample_annotation_count} 个标注")
        
        if sample_data_count == 0:
            no_data_cnt += 1
            if no_data_cnt <= 5:
                print(f"  警告: 样本 {sample_token} 没有样本数据")
        if sample_annotation_count == 0:
            no_annotation_cnt += 1
            if no_annotation_cnt <= 5:
                print(f"  警告: 样本 {sample_token} 没有标注")
    print(f"总共有 {no_data_cnt} 个样本没有样本数据")
    print(f"总共有 {no_annotation_cnt} 个样本没有标注")
    # 检查样本数据是否引用了不存在的样本
    for sd in sample_data:
        if sd["sample_token"] not in sample_tokens:
            print(f"错误: 样本数据 {sd['token']} 引用了不存在的样本 {sd['sample_token']}")
    
    # 检查标注的实例是否存在
    instance_tokens = set(instance["token"] for instance in instances)
    for annotation in sample_annotations:
        instance_token = annotation["instance_token"]
        if instance_token not in instance_tokens:
            print(f"错误: 标注 {annotation['token']} 引用了不存在的实例 {instance_token}")
    
    # 检查实例的标注计数是否与实际匹配
    err_cnt = 0
    for instance in instances:
        instance_token = instance["token"]
        actual_annotation_count = len([ann for ann in sample_annotations if ann["instance_token"] == instance_token])
        if instance["nbr_annotations"] != actual_annotation_count:
            err_cnt += 1
            if err_cnt > 5:
                continue
            print(f"错误: 实例 {instance_token} 的标注计数不一致，记录为 {instance['nbr_annotations']}，实际为 {actual_annotation_count}")
            
    print(f"实例标注计数不一致的实例总数: {err_cnt}")
    
    # 检查样本的"data"字段是否完整
    cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    for sample in samples:
        if "data" not in sample:
            print(f"错误: 样本 {sample['token']} 没有data字段")
        else:
            for cam in cameras:
                if cam not in sample["data"]:
                    print(f"警告: 样本 {sample['token']} 缺少相机 {cam} 的数据")
    
    print("=== 数据一致性检查完成 ===")

def validate_sweeps(sample_data, min_sweeps=1, channel="LIDAR_TOP"):
    """按 sample_token 检查 sweeps；仅统计指定 channel 的非关键帧 bin 记录。"""
    max_print = 5
    print(f"=== Sweeps 数据检查（频道: {channel}，最小 sweeps 数: {min_sweeps}） ===")
    # 按 sample_token 分组
    by_sample = defaultdict(list)
    for sd in sample_data:
        by_sample[sd["sample_token"]].append(sd)

    missing = 0
    missing_details = []
    for s_token, sds in by_sample.items():
        # 找关键帧 lidar
        key = [x for x in sds if x.get("is_key_frame")
               and x.get("fileformat") == "bin"
               and x.get("calibrated_sensor_token")]
        if not key:
            continue
        key_cal = key[0].get("calibrated_sensor_token")
        key_fn = key[0].get("filename", "")

        # 统计 sweeps
        sweeps = [x for x in sds if (not x.get("is_key_frame"))
                  and x.get("fileformat") == "bin"
                  and x.get("calibrated_sensor_token") == key_cal]
        if len(sweeps) < min_sweeps:
            missing += 1
            if len(missing_details) < max_print:
                missing_details.append({
                    "sample_token": s_token,
                    "key_frame": key_fn,
                    "sweep_count": len(sweeps),
                    "sweeps": [sd.get("filename", "") for sd in sweeps]
                })

    print(f"Sweeps 验证: 共 {len(by_sample)} 个样本，缺少 >= {min_sweeps} sweeps 的样本 {missing} 个")
    if missing_details:
        print(f"缺失 sweeps 的样本（最多展示 {max_print} 条）：")
        for i, info in enumerate(missing_details, 1):
            print(f" #{i} sample_token={info['sample_token']}")
            print(f"    key_frame: {info['key_frame']}")
            print(f"    sweep_count: {info['sweep_count']}")
            for sw in info["sweeps"]:
                print(f"      sweep: {sw}")

def main():
    parser = argparse.ArgumentParser(description='Convert custom annotations to NuScenes format')
    parser.add_argument('--input', type=str, required=True, help='Path to input custom annotation JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory for NuScenes format files')
    parser.add_argument('--camera_calib', type=str, required=True, help='Path to camera calibration directory')
    parser.add_argument('--localization', type=str, required=True, help='Path to localization CSV files directory (contains multiple localization.csv files)')
    parser.add_argument('--seq_data', type=str, required=True, help='Path to sequence data directory')
    parser.add_argument('--zero_pose', action='store_true', help='使用零位姿代替从localization.csv计算位姿，可大幅提高处理速度')
    parser.add_argument('--pathmap', type=str, required=True, help='Path to label1st_pathmap.json file')
    parser.add_argument('--use_sweeps', action='store_true', help='为 LIDAR_TOP 生成 sweeps（非关键帧）')
    parser.add_argument('--sweep_window', type=int, default=2, help='每侧可用的 sweeps 数量（按时间距离筛选）')
    parser.add_argument('--sweep_max_dt_us', type=int, default=500_000, help='单个 sweep 与关键帧的最大时间差(微秒)')
    parser.add_argument('--sweep_root', type=str, default=None,
                        help='10Hz sweep 数据根目录，形如 /home/bev_data')
    args = parser.parse_args()
    import gc
    # 提前运行过check_localization_csv.py(含preprocess_localization_and_clean)的话，注释下面这行
    # preprocess_localization_and_clean(args.localization, args.seq_data, args.sweep_root)
    # 加载自定义标注
    print("Loading custom annotations...")
    custom_annotations = load_custom_annotations(args.input)
    custom_annotations = filter_annotations_by_existing_sequences(custom_annotations, 
                                                        args.seq_data)
    custom_annotations = sorted(custom_annotations, \
                                key=lambda x: x.get("timestamp", 0))
    # 加载路径映射
    print("Loading path mapping...")
    with open(args.pathmap, 'r') as f:
        path_mapping = json.load(f)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 创建NuScenes类别
    print("Creating NuScenes categories...")
    categories = create_nuscenes_categories()

    # 获取所有自定义标签
    custom_labels = []
    for frame in custom_annotations:
        custom_labels.extend(frame.get("labels", []))

    # 创建自定义标签到NuScenes类别的映射
    print("Creating custom to NuScenes category mapping...")
    custom_to_nuscenes = create_custom_to_nuscenes_mapping(custom_labels)

    # 创建NuScenes属性
    print("Creating NuScenes attributes...")
    attributes = create_nuscenes_attributes()

    # 创建NuScenes可见性
    print("Creating NuScenes visibility...")
    visibility = create_nuscenes_visibility()

    # 创建NuScenes传感器
    print("Creating NuScenes sensors...")
    sensors = create_nuscenes_sensors()

    # 创建NuScenes校准传感器
    print("Creating NuScenes calibrated sensors...")
    calibrated_sensors = create_nuscenes_calibrated_sensors(sensors, args.camera_calib)

    # 创建NuScenes场景
    print("Creating NuScenes scenes...")
    scenes = create_nuscenes_scene(custom_annotations)

    # 创建NuScenes日志
    print("Creating NuScenes logs...")
    logs = []
    for scene in scenes:
        log = create_nuscenes_log()
        logs.append(log)
        scene["log_token"] = log["token"]

    # 创建NuScenes样本
    print("Creating NuScenes samples...")
    samples = create_nuscenes_samples(custom_annotations, scenes)
    gc.collect()  # 强制垃圾回收
    # 创建NuScenes实例
    print("Creating NuScenes instances...")
    instances, instance_id_to_token = create_nuscenes_instances(custom_annotations, custom_to_nuscenes,\
                                                                 categories)
    gc.collect()  # 强制垃圾回收                                                                 

    # 创建NuScenes样本数据
    print("Creating NuScenes sample data...")
    sample_data_list = create_nuscenes_sample_data_optimized(
        custom_annotations, samples, calibrated_sensors,
        sensors, scenes, path_mapping, args.seq_data,
        zero_pose=args.zero_pose,
        use_sweeps=args.use_sweeps,
        sweep_window=args.sweep_window,
        sweep_max_dt_us=args.sweep_max_dt_us,
        sweep_root=args.sweep_root
    )
    validate_sweeps(sample_data_list, channel="LIDAR_TOP", min_sweeps=1 if args.use_sweeps else 0)
    # create_nuscenes_sample_data(custom_annotations, samples, calibrated_sensors, \
    #                             sensors, scenes, path_mapping, args.seq_data, zero_pose=args.zero_pose)
    gc.collect()  # 强制垃圾回收
    # 创建NuScenes自车位姿
    print("Creating NuScenes ego poses...")
    map_meta_data = {} # 初始化
    
    # 统一位姿生成逻辑（支持 Zero Pose 生成对应时间戳的虚拟位姿）
    ego_poses, map_meta_data = create_nuscenes_ego_poses(samples, custom_annotations, args.localization, path_mapping, zero_pose=args.zero_pose)
    
    # 【新增】安全检查：如果 ego_poses 为空，必须创建一个默认的
    if not ego_poses:
        print("警告: 未生成任何 ego_pose，将创建一个默认零位姿以防止 KeyError")
        default_pose = {
            "token": str(uuid.uuid4()),
            "timestamp": 0,
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0]
        }
        ego_poses.append(default_pose)

    # 通过 timestamp 匹配 sample_data -> ego_pose_token
    ego_timestamp_to_token = {int(ep["timestamp"]): ep["token"] for ep in ego_poses}
    
    # 提取所有有效的时间戳用于快速查找（排序）
    sorted_ego_ts = sorted(ego_timestamp_to_token.keys())
    import bisect

    # 【修改开始】方案 A：过滤掉无法匹配位姿的 sample_data
    valid_sample_data_list = []
    discarded_count = 0

    for sd in sample_data_list:
        sd_ts = sd.get("timestamp", sd.get("file_timestamp", None))
        sd_ts_us = _normalize_to_us_short(sd_ts)
        
        if sd_ts_us is None:
            print(f"警告: SampleData {sd['token']} 时间戳无效，已丢弃")
            discarded_count += 1
            continue

        # 1. 精确匹配
        if sd_ts_us in ego_timestamp_to_token:
            sd["ego_pose_token"] = ego_timestamp_to_token[sd_ts_us]
            valid_sample_data_list.append(sd)
            continue
        
        # 2. 快速最近邻搜索
        idx = bisect.bisect_left(sorted_ego_ts, sd_ts_us)
        
        candidates = []
        if idx < len(sorted_ego_ts):
            candidates.append(sorted_ego_ts[idx])
        if idx > 0:
            candidates.append(sorted_ego_ts[idx-1])
        
        best_token = None
        if candidates:
            # 找差值最小的
            best_ts = min(candidates, key=lambda x: abs(x - sd_ts_us))
            # 检查时间差是否在容忍范围内 (例如 1秒 = 1,000,000 微秒)
            # 对于 39km/h 的车速，100ms 意味着约 1.1 米的位移误差容忍度，
            # 虽然精度高，但容易误杀正常数据，改为150ms
            if abs(best_ts - sd_ts_us) < 150_000: 
                best_token = ego_timestamp_to_token[best_ts]
        
        if best_token is not None:
            sd["ego_pose_token"] = best_token
            valid_sample_data_list.append(sd)
        else:
            # print(f"严重警告: SampleData {sd['token']} (ts={sd_ts_us}) 无法匹配 EgoPose，已丢弃！")
            if sd.get("is_key_frame") and sd.get("channel") == "LIDAR_TOP":
                print(f"!!! 危险: 丢弃了 LiDAR 关键帧! ts={sd_ts_us}")
            discarded_count += 1

    print(f"位姿匹配完成: 保留 {len(valid_sample_data_list)} 条数据，丢弃 {discarded_count} 条无位姿数据")
    sample_data_list = valid_sample_data_list
    # === 重新构建 prev/next 链表 ===
    # 原因：过滤操作删除了中间节点，导致旧的 prev/next 指向了不存在的 token，必须重链
    print("正在修复 SampleData 链表关系 (Re-linking)...")
    
    # 1. 按 calibrated_sensor_token 分组
    sensor_to_sds = defaultdict(list)
    for sd in sample_data_list:
        sensor_to_sds[sd["calibrated_sensor_token"]].append(sd)
        
    # 2. 对每组进行排序和重链
    for sensor_token, sds in sensor_to_sds.items():
        # 按时间戳排序
        sds.sort(key=lambda x: x["timestamp"])
        
        for i, sd in enumerate(sds):
            # 更新 prev
            if i > 0:
                sd["prev"] = sds[i-1]["token"]
            else:
                sd["prev"] = ""
            
            # 更新 next
            if i < len(sds) - 1:
                sd["next"] = sds[i+1]["token"]
            else:
                sd["next"] = ""

    # 创建NuScenes样本标注
    print("Creating NuScenes sample annotations...")
    sample_annotations = create_nuscenes_sample_annotations(
        custom_annotations, samples, instances, instance_id_to_token, custom_to_nuscenes,
        categories, attributes, visibility, path_mapping, args.seq_data,
        ego_poses=ego_poses, calibrated_sensors=calibrated_sensors, sensors=sensors,
        localization_dir=args.localization, 
        use_numba=True
    )
    gc.collect()  # 强制垃圾回收
    # 创建NuScenes地图
    print("Creating NuScenes map...")
    map_data = create_nuscenes_map()
    map_data["log_tokens"] = [l["token"] for l in logs]
    validate_data_consistency(samples, sample_data_list, sample_annotations, instances)
    samples = [{key: value for key, value in sub_dict.items() \
                 if (key!= 'data' and key!= 'anns')} for \
                    sub_dict in samples]
    # 保存所有JSON文件
    print("Saving NuScenes JSON files...")
    with open(os.path.join(args.output, 'category.json'), 'w') as f:
        json.dump(categories, f, indent=2)

    with open(os.path.join(args.output, 'attribute.json'), 'w') as f:
        json.dump(attributes, f, indent=2)

    with open(os.path.join(args.output, 'visibility.json'), 'w') as f:
        json.dump(visibility, f, indent=2)

    with open(os.path.join(args.output, 'sensor.json'), 'w') as f:
        json.dump(sensors, f, indent=2)
    with open(os.path.join(args.output, 'calibrated_sensor.json'), 'w') as f:
        json.dump(calibrated_sensors, f, indent=2)

    with open(os.path.join(args.output, 'scene.json'), 'w') as f:
        json.dump(scenes, f, indent=2)

    with open(os.path.join(args.output, 'log.json'), 'w') as f:
        json.dump(logs, f, indent=2)

    with open(os.path.join(args.output, 'sample.json'), 'w') as f:
        json.dump(samples, f, indent=2)

    with open(os.path.join(args.output, 'instance.json'), 'w') as f:
        json.dump(instances, f, indent=2)

    with open(os.path.join(args.output, 'sample_data.json'), 'w') as f:
        json.dump(sample_data_list, f, indent=2)

    with open(os.path.join(args.output, 'ego_pose.json'), 'w') as f:
        json.dump(ego_poses, f, indent=2)

    with open(os.path.join(args.output, 'sample_annotation.json'), 'w') as f:
        json.dump(sample_annotations, f, indent=2)

    with open(os.path.join(args.output, 'map.json'), 'w') as f:
        json.dump([map_data], f, indent=2)
    # 保存 map_meta.json
    # 这个文件不是 NuScenes 标准格式，但是是你自定义数据的“说明书”
    # 未来接入真地图时，读取这个文件，把 ego_pose 加上 x/y 偏移，就能对齐地图了
    print("Saving Map Metadata...")
    with open(os.path.join(args.output, 'map_meta.json'), 'w') as f:
        json.dump(map_meta_data, f, indent=2)

    print(f"Conversion completed! NuScenes format files saved to {args.output}")

if __name__ == "__main__":
    main()
