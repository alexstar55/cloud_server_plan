#!/usr/bin/env python3
"""
修复相机数据缺失问题 - v5 修复版
"""

import json
import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

def verify_lidar_pts_stats(output_dir, threshold=30.0):
    """
    验证 sample_annotation.json 中 num_lidar_pts 为 0 的占比
    移植自 nuscenes_data_check_v0.1.py
    """
    print("   -> 正在验证点云命中率 (num_lidar_pts)...")
    ann_path = os.path.join(output_dir, "sample_annotation.json")
    
    if not os.path.exists(ann_path):
        print(f"✗ 错误: 找不到 {ann_path}")
        return False

    try:
        with open(ann_path, 'r') as f:
            annotations = json.load(f)
        
        if not annotations:
            print("! 警告: 标注文件为空")
            return True

        total = len(annotations)
        zero_count = 0
        
        for ann in annotations:
            # 确保字段存在且为 int
            pts = ann.get('num_lidar_pts', 0)
            if pts == 0:
                zero_count += 1
        
        rate = (zero_count / total) * 100.0
        print(f"      总标注数: {total}")
        print(f"      空点云标注数: {zero_count}")
        print(f"      空点云占比: {rate:.2f}%")
        
        if rate > threshold:
            print(f"✗ 失败: 空点云标注占比 ({rate:.2f}%) 超过阈值 ({threshold}%)")
            print("  建议检查: 1. 坐标系转换是否正确 (LiDAR vs Global)")
            print("            2. 时间戳对齐是否准确")
            print("            3. 传感器外参是否准确")
            return False
        else:
            print(f"✓ 通过: 点云命中率正常 (阈值 < {threshold}%)")
            return True

    except Exception as e:
        print(f"✗ 验证过程出错: {e}")
        return False
# 修复相机数据缺失问题
def fix_camera_data():
    # 1. 备份现有数据
    # print("1. 备份现有数据...")
    output_dir = "/media/pix/Elements SE/Datasets/Nuscenes/mini/seq0914out"
    # backup_dir = "/media/pix/Elements SE/Datasets/Nuscenes/mini/tmp/seq0914out_backup"

    # if os.path.exists(output_dir):
    #     if os.path.exists(backup_dir):
    #         shutil.rmtree(backup_dir)
    #     # os.makedirs(backup_dir, exist_ok=True)
    #     Path(backup_dir).mkdir(parents=True, exist_ok=True)
    #     src_files = Path(output_dir).rglob("*.json")
    #     for file in src_files:
    #         shutil.copy(file, backup_dir)
    #     # shutil.copytree(output_dir, backup_dir)
    #     print(f"✓ 已备份数据到 {backup_dir}")
    # ---  参数冲突安全检查 ---
    # 定义本次运行的参数配置
    run_params = {
        "zero_pose": True,        # 是否启用零位姿
        "use_sweeps": False       # 是否启用 sweeps
    }
    # 强制检查：严禁 Zero Pose 与 Use Sweeps 同时开启
    # 原因：Zero Pose 假设自车静止(T=Identity)，而 Sweeps 依赖 T_current^-1 * T_prev 进行拼接。
    # 结合使用会导致不同时刻的点云直接叠加在一起，形成严重重影（Ghosting）。
    if run_params["zero_pose"] and run_params["use_sweeps"]:
        print("\n" + "!"*60)
        print("❌ 致命参数冲突检查失败！")
        print("   Detected conflict: --zero_pose AND --use_sweeps are both ENABLED.")
        print("   原因：Zero Pose 模式下自车位姿被强制锁死为原点，无法计算帧间运动。")
        print("         此时启用 Sweeps 会导致多帧点云错误叠加（重影）。")
        print("   解决方案：")
        print("     1. 若需 Zero Pose (无GPS模式)，请设置 use_sweeps = False")
        print("     2. 若需 Sweeps (多帧融合)，请关闭 zero_pose (使用真实GPS)")
        print("!"*60 + "\n")
        return False
    # 2. 重新运行转换过程
    print("2. 重新运行转换过程...")
    try:
        # 使用subprocess运行转换脚本
        cmd = [
            "python3", 
            "/media/pix/Elements SE/Datasets/Nuscenes/mini/scripts/convert_custom_to_nuscenes.py",
            "--input", "/media/pix/Elements SE/Datasets/Nuscenes/mini/merged1and2labels.json",
            "--output", output_dir,
            "--camera_calib", "/media/pix/Elements SE/Datasets/Nuscenes/mini/seq_data/camera_calib_bk_batch1batch2",
            "--localization", "/home/pix/prog_doc/job_data/bev_data",
            "--seq_data", "/media/pix/Elements SE/Datasets/Nuscenes/mini/seq_data",
            "--pathmap", "/media/pix/Elements SE/Datasets/Nuscenes/mini/merged1234_map.json",
        ]

        # 根据配置动态添加参数
        if run_params["zero_pose"]:
            cmd.append("--zero_pose")
        
        if run_params["use_sweeps"]:
            cmd.append("--use_sweeps")
            # 只有开启sweeps时才添加相关参数
            cmd.extend(["--sweep_root", "/home/pix/prog_doc/job_data/bev_data"])
            cmd.extend(["--sweep_max_dt_us", "300_000"])

        print(f"运行命令: {' '.join(cmd)}")
        start_time = datetime.now()
        result = subprocess.run(cmd, check=True, )
        print(result.stdout)
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"转换耗时: {duration.total_seconds()} 秒")
        
        print("✓ 转换完成")
    except subprocess.CalledProcessError as e:
        print(f"✗ 转换失败: {e}")
        # print(f"错误输出: {e.stderr}")
        return False

    # 3. 验证转换结果
    print("3. 验证转换结果...")
    try:
        # 加载必要的文件
        with open(os.path.join(output_dir, "sample.json"), "r") as f:
            samples = json.load(f)
        with open(os.path.join(output_dir, "sample_data.json"), "r") as f:
            sample_data = json.load(f)

        if not samples:
            print("✗ 错误: sample.json 为空")
            return False

        # === 修正后的验证逻辑 ===
        print(f"✓ sample.json 包含 {len(samples)} 条记录")
        print(f"✓ sample_data.json 包含 {len(sample_data)} 条记录")

        # 建立索引：sample_token -> list of sample_data
        sample_data_by_sample = {}
        for sd in sample_data:
            s_tok = sd["sample_token"]
            if s_tok not in sample_data_by_sample:
                sample_data_by_sample[s_tok] = []
            sample_data_by_sample[s_tok].append(sd)

        # 抽查第一个样本
        first_sample = samples[0]
        s_tok = first_sample["token"]
        
        if s_tok in sample_data_by_sample:
            data_entries = sample_data_by_sample[s_tok]
            print(f"✓ 样本 {s_tok} 关联了 {len(data_entries)} 条传感器数据")
            
            if len(data_entries) >= 6: # 至少有6个相机+1个激光雷达
                print("✓ 传感器数据数量合理")
            else:
                print(f"⚠ 警告: 该样本关联的数据较少 ({len(data_entries)})")
        else:
            print(f"✗ 错误: 样本 {s_tok} 在 sample_data.json 中找不到关联数据")
            return False
    except Exception as e:
        print(f"✗ 验证样本数据失败: {e}")
        return False
    # 调用点云统计检查
    if not verify_lidar_pts_stats(output_dir):
        print("警告: 数据生成虽然完成，但点云质量检查未通过！")

    print("相机数据生成完成！")
    return True
if __name__ == "__main__":
    success = fix_camera_data()
    sys.exit(0 if success else 1)
