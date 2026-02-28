import os, re,math
import pandas as pd
import uuid, bisect
import numpy as np
from datetime import datetime
from pathlib import Path
import numba as nb
from collections import defaultdict
from pyproj import Transformer, CRS
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp 
from convert_to_nuscenes_main import extract_english_category

def create_nuscenes_scene(custom_annotations):
    """创建NuScenes场景"""
    scenes = []

    # 按序列分组
    sequences = {}
    for frame in custom_annotations:
        # 从URL中提取序列名称
        info_url = frame.get("info", "")
        if info_url:
            # 提取序列名称，例如从URL中提取"sequence00000"
            parts = info_url.split("/")
            for part in parts:
                if part.startswith("sequence") or \
                        part.startswith("bev_date_"): # 添加对bev_date_开头的序列名称的支持
                    sequence_name = part
                    break
            else:
                # 如果没有找到以"sequence"开头的部分，使用默认值
                sequence_name = "sequence_default"
        else:
            sequence_name = "sequence_default"

        if sequence_name not in sequences:
            sequences[sequence_name] = []
        sequences[sequence_name].append(frame)

    # 为每个序列创建一个场景
    for seq_name, frames in sequences.items():
        scene = {
            "token": str(uuid.uuid4()),
            "name": seq_name,
            "description": f"Converted from sequence {seq_name}",
            "log_token": str(uuid.uuid4()),  # 稍后填充
            "nbr_samples": len(frames),
            "first_sample_token": "",  # 稍后填充
            "last_sample_token": "",  # 稍后填充
        }
        scenes.append(scene)

    return scenes

def create_nuscenes_log():
    """创建NuScenes日志"""
    log = {
        "token": str(uuid.uuid4()),
        "logfile": "custom_log",
        "vehicle": "custom_vehicle",
        "date_captured": datetime.now().strftime("%Y-%m-%d"),
        "location": "custom_location"
    }

    return log

def create_nuscenes_samples(custom_annotations, scenes):
    """创建NuScenes样本"""
    samples = []
    scene_samples = {}  # 存储每个场景的样本

    # 按序列分组
    sequences = {}
    for frame in custom_annotations:
        # 从URL中提取序列名称
        info_url = frame.get("info", "")
        if info_url:
            # 提取序列名称，例如从URL中提取"sequence00000"
            parts = info_url.split("/")
            for part in parts:
                if part.startswith("sequence") or \
                        part.startswith("bev_date_"): # 添加对bev_date_开头的序列名称的支持
                    sequence_name = part
                    break
            else:
                # 如果没有找到以"sequence"开头的部分，使用默认值
                sequence_name = "sequence_default"
        else:
            sequence_name = "sequence_default"

        if sequence_name not in sequences:
            sequences[sequence_name] = []
        sequences[sequence_name].append(frame)

    # 为每个序列的每个帧创建一个样本
    for scene in scenes:
        seq_name = scene["name"]
        scene_token = scene["token"]
        scene_samples[seq_name] = []

        if seq_name in sequences:
            frames = sequences[seq_name]

            # 创建带时间戳的帧列表
            timestamped_frames = []
            for i, frame in enumerate(frames):
                 # 从URL中提取时间戳，如果没有则使用帧索引
                info_url = frame.get("info", "")
                if info_url:
                    # 从URL中提取时间戳并统一为微秒
                    timestamp_str = Path(info_url).stem
                    timestamp = _normalize_to_us(timestamp_str)
                else:
                    # 如果没有URL，使用帧索引生成伪时间戳（微秒）
                    timestamp = i * 100000
                timestamped_frames.append((timestamp, frame, i))

            # 按时间戳排序
            timestamped_frames.sort(key=lambda x: x[0])
            # 过滤时间戳过近的帧 (防止 dt 趋近于 0 导致速度爆表)
            # 最小间隔 50ms (20Hz)
            MIN_FRAME_INTERVAL_US = 50_000 
            filtered_frames = []
            last_ts = -1
            
            for item in timestamped_frames:
                curr_ts = item[0]
                if last_ts == -1 or (curr_ts - last_ts) >= MIN_FRAME_INTERVAL_US:
                    filtered_frames.append(item)
                    last_ts = curr_ts
                else:
                    # 打印警告，方便排查
                    # print(f"警告: 丢弃过近帧 ts={curr_ts}, dt={curr_ts - last_ts} us")
                    pass
            
            timestamped_frames = filtered_frames
            # 创建样本
            for idx, (timestamp, frame, original_idx) in enumerate(timestamped_frames):
                # 如果有重复的时间戳，添加一个小的偏移量确保唯一性
                # (有了上面的过滤，这里其实不太需要了，但保留作为双重保险)
                if idx > 0 and timestamp == timestamped_frames[idx-1][0]:
                    timestamp = timestamped_frames[idx-1][0] + 1000 # 强制加 1ms

                sample = {
                    "token": str(uuid.uuid4()),
                    "timestamp": timestamp,
                    "prev": scene_samples[seq_name][-1]["token"] if idx > 0 else "",
                    "next": "",
                    "scene_token": scene_token,
                    "data": {}  # 初始化数据字段，将在create_nuscenes_sample_data中填充
                }

                if idx > 0:
                    scene_samples[seq_name][-1]["next"] = sample["token"]

                scene_samples[seq_name].append(sample)
                samples.append(sample)

            # 更新场景的第一个和最后一个样本令牌
            if scene_samples[seq_name]:
                scene["first_sample_token"] = scene_samples[seq_name][0]["token"]
                scene["last_sample_token"] = scene_samples[seq_name][-1]["token"]

    return samples

def create_nuscenes_instances(custom_annotations, custom_to_nuscenes, categories):
    """创建NuScenes实例（改进：使用全局唯一的帧标识）"""
    instances = []
    category_name_to_token = {cat["name"]: cat["token"] for cat in categories}

    # 跟踪已经处理的实例
    processed_instances = set()
    # 创建实例ID到实例对象的映射
    id_to_instance = {}
    # 用于调试：跟踪 sequence_id 的分布
    sequence_id_counts = {}
    
    # 【新增】统计被丢弃的标签
    discarded_stats = defaultdict(int)

    for frame in custom_annotations:
        # 从 frame 中获取 sequence_id
        sequence_id = frame.get("_id", "sequence_default")
        
        for label in frame.get("labels", []):
            label_id = label.get("id")
            label_text = str(label.get("label", "")).strip().lower()
            
            # 调试：统计 sequence_id
            sequence_id_counts[sequence_id] = sequence_id_counts.get(sequence_id, 0) + 1
            
            # 确保 _id 存在
            if sequence_id == "sequence_default":
                # 仅打印一次警告防止刷屏，或者保留现状
                pass 
            
            # instance_key 只包含 sequence_id 和 label_id
            instance_key = f"{sequence_id}_{label_id}"
            
            if instance_key in processed_instances:
                inst = id_to_instance.get(instance_key)
                if inst is not None:
                    inst["nbr_annotations"] += 1
                continue

            # 用 label_text 查映射
            nuscenes_category = custom_to_nuscenes.get(label_text)
            
            # 若未找到，实时回退并缓存
            if nuscenes_category is None:
                eng = extract_english_category(label_text)
                mapping = {
                    "bus": "vehicle.bus.rigid",
                    "truck": "vehicle.truck",
                    "motorcycle": "vehicle.motorcycle",
                    "cyclelist": "vehicle.motorcycle",
                    "bicycle": "vehicle.bicycle",
                    "pedestrian": "human.pedestrian.adult",
                    "animal": "animal",
                    "barrier": "movable_object.barrier",
                    "cone": "movable_object.trafficcone",
                    "trailer": "vehicle.trailer",
                    "construction": "vehicle.construction",
                    "constructionvehicle": "vehicle.construction",
                    "car": "vehicle.car",
                    "debris": "movable_object.debris",
                    # 可以在这里添加更多映射
                }
                nuscenes_category = mapping.get(eng)
                
                if nuscenes_category is None:
                    # 【修改】记录丢弃统计，然后跳过（不打印单条日志）
                    discarded_stats[label_text] += 1
                    continue
                
                custom_to_nuscenes[label_text] = nuscenes_category
                # print(f"实例阶段实时映射并缓存: label_text='{label_text}' -> {nuscenes_category}")

            category_token = category_name_to_token.get(nuscenes_category, "")

            # 创建实例
            instance = {
                "token": str(uuid.uuid4()),
                "category_token": category_token,
                "instance_name": label.get("label", "unknown"),
                "nbr_annotations": 1,
                "first_annotation_token": "",
                "last_annotation_token": ""
            }

            # 存储实例对象
            id_to_instance[instance_key] = instance
            instances.append(instance)
            processed_instances.add(instance_key) # 别忘了标记已处理

    # 调试输出
    print("\n=== 序列ID统计 ===")
    for seq_id, count in sorted(sequence_id_counts.items()):
        # 简化输出，只打印前几个或总数
        pass 
    print(f"总共 {len(processed_instances)} 个唯一实例")

    # 【新增】在函数结束时打印丢弃类别汇总
    if discarded_stats:
        print("\n" + "!"*60)
        print("【注意】以下原始标签因无法映射到NuScenes类别而被丢弃（未生成实例）：")
        print(f"{'原始标签 (Raw Label)':<40} | {'丢弃实例数':<10}")
        print("-" * 55)
        # 按丢弃次数降序排列
        for label, count in sorted(discarded_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{label:<40} | {count:<10}")
        print("-" * 55)
        print("提示：如果上述列表中包含有效类别（非 unknown），请在 create_nuscenes_instances 的 mapping 字典中添加映射。")
        print("!"*60 + "\n")
    else:
        print("\n[Info] 所有标签均成功映射，无丢弃实例。\n")
    
    # 添加实例ID到实例令牌的映射
    instance_key_to_token = {instance_key: instance["token"] for instance_key, \
                             instance in id_to_instance.items()}

    return instances, instance_key_to_token

def process_image_url(img_url, path_mapping, seq_data):
    # 从URL中提取sequence和文件路径
    # 示例URL: https://molar-app-saas.oss-cn-hangzhou.aliyuncs.com/NjhkMjE4MzU5YWQxZDU5ZTg0MmFiY2Fh/20250923115222/sequence00000/front/1757783927.700042.jpg
    # 提取sequence及之后的路径
    # match = re.search(r"sequence[0-9]+/.+$", img_url) # 修改正则表达式以匹配新的URL格式
    match = re.search(r"(sequence[0-9]+|bev_date_[0-9]+_sequence[0-9]+)/.+$", img_url)
    if not match:
        print(f"警告: 无法从URL中提取sequence路径: {img_url}")
        return None
    
    relative_path = match.group(0)  # sequence00000/front/1757783927.700042.jpg 或 bev_date_202509190139_sequence00018/front/1758217589.500049.jpg
    # 分离sequence和剩余路径
    sequence, *path_parts = relative_path.split("/")
    
    # 检查sequence是否在path_mapping中
    if sequence not in path_mapping:
        print(f"警告: sequence {sequence} 不在path_mapping中")
        return None
    
    try:
        # 构建完整路径
        mapped_sequence = path_mapping[sequence]
        
        # 检查mapped_sequence是否已经包含seq_data路径
        if seq_data:
            # 确保seq_data不以斜杠结尾
            seq_data = seq_data.rstrip('/')
            seq_data_name = os.path.basename(seq_data) # 获取最后一级目录名，如 "v1.0-trainval"
            
            # 检查mapped_sequence是否已经包含seq_data (绝对路径或完整前缀)
            if mapped_sequence.startswith(seq_data + '/'):
                # 已经包含seq_data，直接使用mapped_sequence
                full_path = os.path.join(mapped_sequence, *path_parts)
            
            elif mapped_sequence.startswith('seq_data/'):
                # mapped_sequence以seq_data开头，但不包含完整的seq_data路径
                # 替换seq_data为完整的seq_data路径
                mapped_sequence = mapped_sequence.replace('seq_data/', seq_data + '/', 1)
                full_path = os.path.join(mapped_sequence, *path_parts)
            
            # 【新增修复】检查 mapped_sequence 是否以 seq_data 的末尾目录名开头，如果是，则去重
            # 解决 map="v1.0-trainval/bev..." 且 seq_data=".../v1.0-trainval" 导致的路径重复问题
            elif mapped_sequence.startswith(seq_data_name + '/'):
                # 去掉映射路径中的重复前缀
                clean_mapped = mapped_sequence[len(seq_data_name)+1:]
                full_path = os.path.join(seq_data, clean_mapped, *path_parts)
                
            else:
                # 不包含seq_data，添加seq_data
                full_path = os.path.join(seq_data, mapped_sequence, *path_parts)
        else:
            # 没有提供seq_data，直接使用mapped_sequence
            full_path = os.path.join(mapped_sequence, *path_parts)
        
        # 标准化路径，使用正斜杠
        full_path = full_path.replace('\\', '/')
        if not os.path.exists(full_path):
            print(f"警告: 图像文件不存在: {full_path}")
            return None
        return full_path
    except Exception as e:
        print(f"错误: 处理图像路径时出错: {e}")
        print(f"原始URL: {img_url}")
        return None

def _extract_bev_seq_from_path(path: Path):
    """
    从路径中解析 (bev_date, sequence)：
      - 形式1: .../bev_date_xxx/sequence00000/...
      - 形式2: .../bev_date_xxx_sequence00000/...
    """
    bev_date, seq = None, None
    parts = list(path.parts)
    for i, p in enumerate(parts):
        if p.startswith("bev_date_"):
            bev_date = p
            # 形式1
            if i + 1 < len(parts) and parts[i + 1].startswith("sequence"):
                seq = parts[i + 1]
            else:
                # 形式2（同级合并）
                m = re.match(r"(bev_date_\d+)_?(sequence\d+)", p)
                if m:
                    bev_date, seq = m.group(1), m.group(2)
            break
    if seq is None:
        # 尝试在后续部分找到 sequence
        for p in parts:
            if p.startswith("sequence"):
                seq = p
                break
    return bev_date, seq

def create_nuscenes_sample_data_optimized(custom_annotations, samples, calibrated_sensors, 
                                        sensors, scenes, path_mapping=None, seq_data=None, 
                                        zero_pose=False, use_sweeps=False, sweep_window=3, 
                                        sweep_max_dt_us=500_000, sweep_root=None):
    """优化版的单线程样本数据创建"""
    sample_data_list = []
    processed_samples = set()  # 避免同一个 sample 重复生成数据
    # 预先计算所有映射关系
    channel_to_calibrated_token = {}
    for cs in calibrated_sensors:
        for sensor in sensors:
            if sensor["token"] == cs["sensor_token"]:
                channel = sensor["channel"]
                channel_to_calibrated_token[channel] = cs["token"]
                break

    zero_pose_token = str(uuid.uuid4()) if zero_pose else None

    # ---------- 新增：预扫描可用的 (bev_date, sequence) 并准备忽略/打印集 ----------
    allowed_pairs = None
    printed_skip_bev = set()
    printed_skip_pair = set()

    if use_sweeps and sweep_root:
        used_pairs = set()    # 标注里实际使用到的 (bev_date, seq)
        sweep_pairs = set()   # sweep_root 中存在的 (bev_date, seq)

        # 1) 标注帧里提取 (bev_date, seq)
        for frame in custom_annotations:
            info_url = frame.get("info", "")
            if not info_url:
                continue
            bev, seq = _extract_bev_seq_from_path(Path(info_url))
            if bev and seq:
                used_pairs.add((bev, seq))

        # 2) 扫描 sweep_root 下存在的 (bev_date, seq)
        root_path = Path(sweep_root)
        if root_path.exists():
            for d in root_path.iterdir():
                if not d.is_dir():
                    continue
                if d.name.startswith("bev_date_"):
                    # 形式1: bev_date_xxx/sequence00000
                    seq_dirs = [c for c in d.iterdir() if c.is_dir() and c.name.startswith("sequence")]
                    if seq_dirs:
                        for sd in seq_dirs:
                            sweep_pairs.add((d.name, sd.name))
                    else:
                        # 形式2: bev_date_xxx_sequence00000
                        bev2, seq2 = _extract_bev_seq_from_path(d)
                        if bev2 and seq2:
                            sweep_pairs.add((bev2, seq2))

        # 3) 交集
        allowed_pairs = used_pairs & sweep_pairs
        unused_pairs = sweep_pairs - allowed_pairs
        unused_bev = {p[0] for p in unused_pairs}

        print(f"[sweeps] 标注用到的 (bev_date, seq): {len(used_pairs)}")
        print(f"[sweeps] sweep_root 存在的 (bev_date, seq): {len(sweep_pairs)}")
        print(f"[sweeps] 实际用于 sweeps 的交集: {len(allowed_pairs)}")
        if unused_bev:
            print("[sweeps] 下列 bev_date_* 在 sweep_root 中存在但未被标注使用，将被忽略：")
            for b in sorted(unused_bev):
                print(f"  - {b}")
        if unused_pairs:
            print("[sweeps] 下列 (bev_date, sequence) 未被使用，将被忽略：")
            for b, s in sorted(unused_pairs):
                print(f"  - {b}/{s}")
        
    def _collect_sweeps_from_dir(dir_path: Path, key_ts_us: int, pattern="*.bin"):
        if not dir_path or not dir_path.exists():
            return []
        sweeps = []
        for f in dir_path.glob(pattern):
            ts_us = _normalize_to_us(f.stem)
            if ts_us is None or ts_us == key_ts_us:
                continue
            dt = abs(ts_us - key_ts_us)
            if dt <= sweep_max_dt_us:
                sweeps.append((dt, ts_us, f))
        sweeps.sort(key=lambda x: x[0])
        return sweeps[:sweep_window]
    
    cam_dir_to_channel = {
        "front": "CAM_FRONT",
        "front_left": "CAM_BACK_LEFT",
        "front_right": "CAM_BACK_RIGHT",
        "rear": "CAM_BACK",
        "rear_left": "CAM_FRONT_LEFT",
        "rear_right": "CAM_FRONT_RIGHT",
    }
    # 使用更高效的数据结构
    scene_to_samples = {}
    for sample in samples:
        scene_token = sample.get("scene_token", "")
        if scene_token not in scene_to_samples:
            scene_to_samples[scene_token] = []
        scene_to_samples[scene_token].append(sample)

    scene_name_to_scene = {scene["name"]: scene for scene in scenes}
    
    # 预处理：按场景分组自定义标注
    scene_to_frames = {}
    for frame in custom_annotations:
        info_url = frame.get("info", "")
        sequence_name = "sequence_default"
        if info_url:
            parts = info_url.split("/")
            for part in parts:
                if part.startswith("sequence") or \
                        part.startswith("bev_date_"): # 添加对bev_date_开头的序列名称的支持
                    sequence_name = part
                    break
        
        # 找到对应的场景
        if sequence_name in scene_name_to_scene:
            scene_token = scene_name_to_scene[sequence_name]["token"]
            if scene_token not in scene_to_frames:
                scene_to_frames[scene_token] = []
            scene_to_frames[scene_token].append(frame)

    # 处理每个场景
    processed_count = 0
    total_scenes = len(scene_to_frames)
    
    for scene_token, frames in scene_to_frames.items():
        if scene_token not in scene_to_samples:
            continue
            
        scene_samples = scene_to_samples[scene_token]
        processed_count += 1
        print(f"处理场景 {processed_count}/{total_scenes}: {len(frames)} 帧, {len(scene_samples)} 样本")
        
        # 创建时间戳映射
        frame_timestamps = {}
        for frame in frames:
            info_url = frame.get("info", "")
            if info_url:
                timestamp_str = Path(info_url).stem
                timestamp = _normalize_to_us(timestamp_str)
            else:
                timestamp = 0
            frame_timestamps[timestamp] = frame

        # 使用样本的timestamp字典
        sample_timestamps = {sample["timestamp"]: sample for sample in scene_samples}

        # 改为最近邻匹配（容忍时间差阈值）
        tolerance_us = 500_000  # 0.5s，可按需调整
        matched_pairs = []
        sample_ts_list = sorted(sample_timestamps.keys())
        for f_ts, frame in frame_timestamps.items():
            # 找最接近的 sample_ts
            best_sample_ts = None
            best_diff = None
            for s_ts in sample_ts_list:
                diff = abs(s_ts - f_ts)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_sample_ts = s_ts
            if best_diff is not None and best_diff <= tolerance_us:
                matched_pairs.append((f_ts, best_sample_ts))
            else:
                # 未匹配到合适 sample，则跳过或降级处理
                print(f"警告: 未找到匹配 sample for frame_ts={f_ts} (best diff {best_diff}), 跳过该帧")
                continue

        for frame_ts, sample_ts in matched_pairs:
            frame = frame_timestamps[frame_ts]
            sample = sample_timestamps[sample_ts]
            # 跳过已处理过的 sample（防止一个 sample 匹配到多个 frame）
            if sample["token"] in processed_samples:
               continue
            processed_samples.add(sample["token"])
            # 处理点云数据
            info_url = frame.get("info", "")
            if info_url and path_mapping is not None:
                sequence_name = os.path.basename(os.path.dirname(os.path.dirname(info_url)))
                if sequence_name in path_mapping:
                    new_path = path_mapping[sequence_name]
                    filename = os.path.basename(info_url)
                    if seq_data and not new_path.startswith(seq_data):
                        info_url = os.path.join(Path(seq_data).parent, new_path, filename).replace("\\", "/")
                    else:
                        info_url = os.path.join(new_path, filename).replace("\\", "/")
            
            # 创建点云样本数据
            if "lidar" in Path(info_url).parent.name:
                filename = str(Path(info_url).parent/(Path(info_url).stem + ".bin"))
            else:
                filename = str(Path(info_url).parent/"lidar"/(Path(info_url).stem + ".bin"))

            lidar_timestamp = _normalize_to_us(Path(info_url).stem)

            lidar_sample_data = {
                "token": str(uuid.uuid4()),
                "sample_token": sample["token"],
                "ego_pose_token": zero_pose_token if zero_pose else "",
                "calibrated_sensor_token": channel_to_calibrated_token.get("LIDAR_TOP", ""),
                "filename": filename,
                "fileformat": "bin",
                "width": 0,
                "height": 0,
                "timestamp": lidar_timestamp,
                "is_key_frame": True,
                "channel": "LIDAR_TOP",         
                "sensor_modality": "lidar",
                "next": "",
                "prev": ""
            }

            sample_data_list.append(lidar_sample_data)
            if "data" not in sample:
                sample["data"] = {}
            sample["data"]["LIDAR_TOP"] = lidar_sample_data["token"]
            # 解析 bev_date 与 sequence
            bev_date, seq_name = _extract_bev_seq_from_path(Path(info_url))
            # 关键帧 lidar 路径已确定：filename

            # ===== sweeps: LIDAR =====
            if use_sweeps:
                # 若未在允许集合，则跳过 sweeps
                if allowed_pairs is not None and (bev_date, seq_name) not in allowed_pairs:
                    if bev_date and bev_date not in printed_skip_bev:
                        print(f"[sweeps] 跳过 bev_date={bev_date}，未在标注/seq_data 中使用")
                        printed_skip_bev.add(bev_date)
                    if bev_date and seq_name and (bev_date, seq_name) not in printed_skip_pair:
                        print(f"[sweeps] 跳过 pair={bev_date}/{seq_name}")
                        printed_skip_pair.add((bev_date, seq_name))
                else:
                    # 优先使用 sweep_root 下的路径
                    sweep_lidar_dir = None
                    if sweep_root and bev_date and seq_name:
                        # 形式1: bev_date_xxx/sequence00000/lidar
                        cand1 = Path(sweep_root)/bev_date/seq_name/"lidar"
                        # 形式2: bev_date_xxx_sequence00000/lidar
                        cand2 = Path(sweep_root)/(f"{bev_date}_{seq_name}")/"lidar"
                        sweep_lidar_dir = cand1 if cand1.exists() else (cand2 if cand2.exists() else None)
                    if not (sweep_lidar_dir and sweep_lidar_dir.exists()):
                        sweep_lidar_dir = Path(filename).parent  # 回退到关键帧所在目录

                    sweeps = _collect_sweeps_from_dir(sweep_lidar_dir, lidar_timestamp, pattern="*.bin")
                    for _, ts_us, bin_file in sweeps:
                        sweep_sd = {
                            "token": str(uuid.uuid4()),
                            "sample_token": sample["token"],
                            "ego_pose_token": zero_pose_token if zero_pose else "",
                            "calibrated_sensor_token": channel_to_calibrated_token.get("LIDAR_TOP", ""),
                            "filename": str(bin_file),
                            "fileformat": "bin",
                            "width": 0,
                            "height": 0,
                            "timestamp": ts_us,
                            "is_key_frame": False,
                            "channel": "LIDAR_TOP",
                            "sensor_modality": "lidar",
                            "next": "",
                            "prev": ""
                        }
                        sample_data_list.append(sweep_sd)

                    # ===== sweeps: CAM（同样用两种目录形式尝试）=====
                    if sweep_root and bev_date and seq_name:
                        cand_cam1 = Path(sweep_root)/bev_date/seq_name
                        cand_cam2 = Path(sweep_root)/(f"{bev_date}_{seq_name}")
                        cam_root = cand_cam1 if cand_cam1.exists() else (cand_cam2 if cand_cam2.exists() else Path(info_url).parent.parent)
                    else:
                        cam_root = Path(info_url).parent.parent
                    for cam_dir_name, channel in cam_dir_to_channel.items():
                        cam_dir = cam_root/cam_dir_name
                        if not cam_dir.exists():
                            continue
                        cam_sweeps = _collect_sweeps_from_dir(cam_dir, lidar_timestamp, pattern="*.jpg")
                        for _, ts_us, img_file in cam_sweeps:
                            img_sd = {
                                "token": str(uuid.uuid4()),
                                "sample_token": sample["token"],
                                "ego_pose_token": zero_pose_token if zero_pose else "",
                                "calibrated_sensor_token": channel_to_calibrated_token.get(channel, ""),
                                "filename": str(img_file),
                                "fileformat": "jpg",
                                "width": 1920,
                                "height": 1080,
                                "timestamp": ts_us,
                                "is_key_frame": False,
                                "next": "",
                                "prev": ""
                            }
                            sample_data_list.append(img_sd)
            # 批量处理图像数据
            img_info = frame.get("imgInfo", [])
            for img_url in img_info:
                processed_path = process_image_url(img_url, path_mapping, seq_data)
                if not processed_path:
                    continue
                    
                # 确定相机通道
                if "front" in img_url.lower():
                    if "left" in img_url.lower():
                        channel = "CAM_BACK_LEFT"
                    elif "right" in img_url.lower():
                        channel = "CAM_BACK_RIGHT"
                    else:
                        channel = "CAM_FRONT"
                elif "rear" in img_url.lower() or "back" in img_url.lower():
                    if "left" in img_url.lower():
                        channel = "CAM_FRONT_LEFT"
                    elif "right" in img_url.lower():
                        channel = "CAM_FRONT_RIGHT"
                    else:
                        channel = "CAM_BACK"
                else:
                    channel = "CAM_FRONT"
                
                # img_ts = _normalize_to_us(Path(img_url).stem)
                # 使用图像自身文件名时间戳，单位微秒
                img_ts = _normalize_to_us(Path(img_url).stem) # 使用样本关键帧时间戳，保证与标注/ego_pose对齐
                img_sample_data = {
                    "token": str(uuid.uuid4()),
                    "sample_token": sample["token"],
                    "ego_pose_token": zero_pose_token if zero_pose else "",
                    "calibrated_sensor_token": channel_to_calibrated_token.get(channel, ""),
                    "filename": processed_path,
                    "fileformat": "jpg",
                    "width": 1920,
                    "height": 1080,
                    "timestamp": img_ts,
                    "is_key_frame": True,
                    "channel": channel,
                    "sensor_modality": "camera",
                    "next": "",
                    "prev": ""
                }
                
                sample_data_list.append(img_sample_data)
                sample["data"][channel] = img_sample_data["token"]
    # ...after building sample_data_list, 为每个通道补 prev/next ...
    def _link_prev_next(ch_token):
        ch_sds = [sd for sd in sample_data_list
                if sd.get("calibrated_sensor_token") == ch_token]
        ch_sds.sort(key=lambda x: x["timestamp"])
        for i, sd in enumerate(ch_sds):
            sd["prev"] = ch_sds[i-1]["token"] if i > 0 else ""
            sd["next"] = ch_sds[i+1]["token"] if i < len(ch_sds)-1 else ""

    for channel, cal_token in channel_to_calibrated_token.items():
        if cal_token:
            _link_prev_next(cal_token)
    print(f"样本数据处理完成: 总共处理了 {len(processed_samples)} 个唯一样本，生成了 {len(sample_data_list)} 个样本数据条目")
    # print(f"样本数据处理完成: 总共处理了 {len(processed_samples)} 个唯一样本，生成了 {len(sample_data_list)} 个样本数据条目")
    return sample_data_list
def create_nuscenes_sample_data(custom_annotations, samples, calibrated_sensors, sensors, scenes, \
                                path_mapping=None, seq_data=None, zero_pose=False):
    """创建NuScenes样本数据 - 改进版，使用最近邻匹配和真实timestamp"""
    sample_data_list = []
    processed_samples = set()  # 跟踪已处理的样本token
    
    if path_mapping is not None:
        print("Using path mapping to convert file paths...")

    # 创建传感器通道到校准传感器令牌的映射
    channel_to_calibrated_token = {}
    for cs in calibrated_sensors:
        for sensor in sensors:
            if sensor["token"] == cs["sensor_token"]:
                channel = sensor["channel"]
                channel_to_calibrated_token[channel] = cs["token"]
                break

    # 如果使用零位姿，创建一个固定的ego_pose_token
    zero_pose_token = None
    if zero_pose:
        zero_pose_token = str(uuid.uuid4())
        print(f"使用零位姿模式，所有样本数据将使用同一个ego_pose_token: {zero_pose_token}")

    # 按序列分组自定义标注和样本
    sequences = {}
    for frame in custom_annotations:
        # 从URL中提取序列名称
        info_url = frame.get("info", "")
        if info_url:
            # 提取序列名称，例如从URL中提取"sequence00000"
            parts = info_url.split("/")
            for part in parts:
                if part.startswith("sequence") or \
                        part.startswith("bev_date_"): # 添加对bev_date_开头的序列名称的支持
                    sequence_name = part
                    break
            else:
                # 如果没有找到以"sequence"开头的部分，使用默认值
                sequence_name = "sequence_default"
        else:
            sequence_name = "sequence_default"

        if sequence_name not in sequences:
            sequences[sequence_name] = []
        sequences[sequence_name].append(frame)

    # 从scene中获取序列样本映射
    sequence_samples = {}
    for sample in samples:
        scene_token = sample.get("scene_token", "")
        for scene in scenes:
            if scene["token"] == scene_token:
                seq_name = scene["name"]
                if seq_name not in sequence_samples:
                    sequence_samples[seq_name] = []
                sequence_samples[seq_name].append(sample)
                break

    # 【改进】为每个序列的每个样本创建样本数据（使用最近邻匹配）
    tolerance_us = 500_000  # 0.5s 容忍度
    
    for seq_name, frames in sequences.items():
        if seq_name not in sequence_samples:
            print(f"警告: 序列 {seq_name} 没有对应的样本，跳过")
            continue

        print(f"处理序列: {seq_name}, 包含 {len(frames)} 帧和 {len(sequence_samples[seq_name])} 个样本")
        
        # 【改进】创建时间戳映射（frame timestamp -> frame）
        frame_timestamps = {}
        for frame in frames:
            info_url = frame.get("info", "")
            if info_url:
                timestamp_str = Path(info_url).stem
                timestamp = _normalize_to_us(timestamp_str)
            else:
                timestamp = 0
            frame_timestamps[timestamp] = frame

        # 【改进】创建样本时间戳列表（用于最近邻查找）
        sample_timestamps = {}
        for sample in sequence_samples[seq_name]:
            sample_timestamps[sample["timestamp"]] = sample
        
        sample_ts_list = sorted(sample_timestamps.keys())
        
        # 【改进】使用最近邻匹配而不是严格的一一对应
        matched_pairs = []
        for f_ts, frame in frame_timestamps.items():
            # 找最接近的 sample_ts
            best_sample_ts = None
            best_diff = None
            
            for s_ts in sample_ts_list:
                diff = abs(s_ts - f_ts)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_sample_ts = s_ts
            
            # 检查差距是否在容忍范围内
            if best_diff is not None and best_diff <= tolerance_us:
                matched_pairs.append((f_ts, best_sample_ts, frame))
            else:
                print(f"警告: 序列 {seq_name} 中 frame_ts={f_ts} 未找到合适的 sample 匹配 (best diff {best_diff}), 跳过该帧")

        print(f"  成功匹配 {len(matched_pairs)} 个 frame-sample 对")
        
        # 【改进】按匹配对创建样本数据
        for frame_ts, sample_ts, frame in matched_pairs:
            sample = sample_timestamps[sample_ts]
            
            # 检查样本是否已处理
            sample_token = sample["token"]
            if sample_token in processed_samples:
                print(f"警告: 样本 {sample_token} 已被处理，跳过重复处理")
                continue
            processed_samples.add(sample_token)

            # ========== 处理点云数据 ==========
            info_url = frame.get("info", "")
            if info_url:
                # 如果有路径映射，则转换路径
                if path_mapping is not None:
                    sequence_name = os.path.basename(os.path.dirname(os.path.dirname(info_url)))
                    if sequence_name in path_mapping:
                        new_path = path_mapping[sequence_name]
                        filename = os.path.basename(info_url)
                        
                        if seq_data and not new_path.startswith(seq_data):
                            info_url = os.path.join(Path(seq_data).parent, new_path, filename).replace("\\", "/")
                        else:
                            info_url = os.path.join(new_path, filename).replace("\\", "/")
            
            # 提取点云文件名
            if "lidar" in Path(info_url).parent.name:
                filename = str(Path(info_url).parent / (Path(info_url).stem + ".bin"))
            else:
                filename = str(Path(info_url).parent / "lidar" / (Path(info_url).stem + ".bin"))

            # 【改进】使用点云文件的真实时间戳（微秒）
            lidar_timestamp = _normalize_to_us(Path(info_url).stem)

            # 创建点云样本数据
            lidar_sample_data = {
                "token": str(uuid.uuid4()),
                "sample_token": sample["token"],
                "ego_pose_token": zero_pose_token if zero_pose else "",
                "calibrated_sensor_token": channel_to_calibrated_token.get("LIDAR_TOP", ""),
                "filename": filename,
                "fileformat": "bin",
                "width": 0,
                "height": 0,
                "timestamp": lidar_timestamp,  # 【改进】使用 lidar 的真实时间戳
                "is_key_frame": True,
                "channel": "LIDAR_TOP",  
                "sensor_modality": "lidar", 
                "next": "",
                "prev": ""
            }

            sample_data_list.append(lidar_sample_data)
            if "data" not in sample:
                sample["data"] = {}
            sample["data"]["LIDAR_TOP"] = lidar_sample_data["token"]

            # ========== 处理图像数据 ==========
            img_info = frame.get("imgInfo", [])
            if not img_info:
                print(f"警告: 样本 {sample_token} 没有图像信息")
            
            for j, img_url in enumerate(img_info):
                try:
                    processed_path = process_image_url(img_url, path_mapping, seq_data)
                    if not processed_path:
                        print(f"警告: 跳过无效的图像URL: {img_url}")
                        continue
                    
                    # 确定相机通道
                    if "front" in img_url.lower():
                        if "left" in img_url.lower():
                            channel = "CAM_BACK_LEFT"
                        elif "right" in img_url.lower():
                            channel = "CAM_BACK_RIGHT"
                        else:
                            channel = "CAM_FRONT"
                    elif "rear" in img_url.lower() or "back" in img_url.lower():
                        if "left" in img_url.lower():
                            channel = "CAM_FRONT_LEFT"
                        elif "right" in img_url.lower():
                            channel = "CAM_FRONT_RIGHT"
                        else:
                            channel = "CAM_BACK"
                    else:
                        channel = "CAM_FRONT"
                    
                    # 【改进】使用图像文件的真实时间戳（微秒）
                    img_ts = _normalize_to_us(Path(img_url).stem)
                    
                    # 创建图像样本数据
                    img_sample_data = {
                        "token": str(uuid.uuid4()),
                        "sample_token": sample["token"],
                        "ego_pose_token": zero_pose_token if zero_pose else "",
                        "calibrated_sensor_token": channel_to_calibrated_token.get(channel, ""),
                        "filename": processed_path,
                        "fileformat": "jpg",
                        "width": 1920,
                        "height": 1080,
                        "timestamp": img_ts,  # 【改进】使用图像的真实时间戳（微秒）
                        "is_key_frame": True,
                        "channel": channel,
                        "sensor_modality": "camera",
                        "next": "",
                        "prev": ""
                    }
                    
                    sample_data_list.append(img_sample_data)
                    if "data" not in sample:
                        sample["data"] = {}
                    sample["data"][channel] = img_sample_data["token"]
                    
                except Exception as e:
                    print(f"错误: 处理图像数据时出错: {e}")
                    print(f"图像URL: {img_url}")
                    continue

    print(f"样本数据处理完成: 总共处理了 {len(processed_samples)} 个唯一样本，生成了 {len(sample_data_list)} 个样本数据条目")
    return sample_data_list

def create_nuscenes_map():
    """创建NuScenes地图"""
    map = {
        "token": str(uuid.uuid4()),
        "log_tokens": [],  # 稍后填充
        "category": "semantic_prior",
        "filename": "custom_map.json"
    }

    return map
def _normalize_to_us(ts):
    """
    统一把不同格式的时间戳转换为微秒（int）。
    规则：
      - 小于 1e11：视为秒， * 1e6
      - 1e11 <= ts < 1e14：视为毫秒， * 1e3
      - >=1e14：视为已经是微秒，直接取整
    支持字符串、float、int。
    """
    if ts is None:
        return None
    try:
        t = float(ts)
    except Exception:
        # 如果是类似 "1757783927.700042.jpg" 的 file-stem，需要先提取数字
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
    
def transform_to_global(local_pos, ego_pose):
    """将局部坐标(x,y,z)转为全局坐标"""
    # NuScenes rotation is [w, x, y, z]
    rot = ego_pose["rotation"]
    trans = ego_pose["translation"]
    
    r = R.from_quat([rot[1], rot[2], rot[3], rot[0]])
    global_pos = r.apply(local_pos) + np.array(trans)
    return global_pos

def calculate_global_velocity(custom_annotations, sequence_id, label_id, current_frame_idx, timestamp_to_ego_pose):
    """
    计算全局绝对速度 (Global Absolute Velocity)
    适用于 Zero Pose 和 Standard NuScenes 格式
    """
    # 1. 搜索当前帧及下一帧
    # 为了效率，只在附近搜索
    search_range = 5
    start_idx = max(0, current_frame_idx - 1)
    end_idx = min(len(custom_annotations), current_frame_idx + search_range)
    
    curr_frame_data = None
    next_frame_data = None
    
    # 提取当前帧信息
    curr_anno = custom_annotations[current_frame_idx]
    if curr_anno.get("_id") != sequence_id: return np.array([0.0, 0.0])
    
    # 获取当前帧的时间戳 (微秒)
    info_url = curr_anno.get("info", "")
    curr_ts = _normalize_to_us(Path(info_url).stem) if info_url else current_frame_idx * 100000
    
    # 获取当前帧的局部坐标
    curr_local_pos = None
    for label in curr_anno.get("labels", []):
        if label.get("id") == label_id:
            p = label.get("points", [])
            if len(p) >= 3: curr_local_pos = np.array(p[:3])
            break
            
    if curr_local_pos is None: return np.array([0.0, 0.0])

    # 寻找下一帧 (同一个ID)
    for i in range(current_frame_idx + 1, end_idx):
        frame = custom_annotations[i]
        if frame.get("_id") != sequence_id: continue
        
        # 找时间戳
        info_url_next = frame.get("info", "")
        next_ts = _normalize_to_us(Path(info_url_next).stem) if info_url_next else i * 100000
        
        # 找点
        for label in frame.get("labels", []):
            if label.get("id") == label_id:
                p = label.get("points", [])
                if len(p) >= 3:
                    next_frame_data = {
                        'ts': next_ts,
                        'local_pos': np.array(p[:3])
                    }
                break
        if next_frame_data: break
    
    if not next_frame_data: return np.array([0.0, 0.0])

    # 2. 获取 Ego Pose
    # 注意：timestamp_to_ego_pose 的 key 必须是微秒 int
    pose_curr = timestamp_to_ego_pose.get(curr_ts)
    pose_next = timestamp_to_ego_pose.get(next_frame_data['ts'])
    
    # 如果找不到精确匹配，尝试最近邻 (容错)
    if pose_curr is None:
        # 简易查找逻辑，实际建议在外部构建好精确映射
        return np.array([0.0, 0.0]) 
    if pose_next is None:
        return np.array([0.0, 0.0])

    # 3. 转全局坐标
    g_curr = transform_to_global(curr_local_pos, pose_curr)
    g_next = transform_to_global(next_frame_data['local_pos'], pose_next)
    
    # 4. 计算全局速度
    dt = (next_frame_data['ts'] - curr_ts) / 1e6
    if dt < 0.02: return np.array([0.0, 0.0]) # 避免 dt 过小
    
    v_global = (g_next - g_curr) / dt
    
    # 5. 阈值过滤 (例如 > 40m/s 视为异常)
    speed = np.linalg.norm(v_global[:2])
    if speed > 40.0:
        return np.array([0.0, 0.0])
        
    # 对于 Zero Pose，因为 Ego Pose 全是 0，这里的 transform_to_global 
    # 实际上就是 transform_to_local (没变)。
    # 但因为我们传入的是“真实的 Ego Pose” (见下文调用处的修改)，
    # 所以这里算出的就是真实的绝对速度。
    
    return v_global[:2]
    
def load_localization_data(localization_path):
    # 加载自车位姿数据 - 兼容不同时间戳单位，支持 UTM 转换和 Heading
    try:
        tmp = pd.read_csv(localization_path, nrows=1, on_bad_lines='skip')
    except Exception as e:
        print(f"Warning: Failed to read header of {localization_path}: {e}")
        return {}

    # 1. 确定时间戳列
    if "timestamp_us" in tmp.columns:
        ts_col = "timestamp_us"
        ts_unit = "us"
    elif "timestamp_ms" in tmp.columns:
        ts_col = "timestamp_ms"
        ts_unit = "ms"
    elif "timestamp_s" in tmp.columns:
        ts_col = "timestamp_s"
        ts_unit = "s"
    else:
        if len(tmp.columns) > 0 and "timestamp" in tmp.columns[0].lower():
             ts_col = tmp.columns[0]
             if "us" in ts_col: ts_unit = "us"
             elif "ms" in ts_col: ts_unit = "ms"
             else: ts_unit = "s"
        else:
            print(f"Warning: localization.csv 缺少 timestamp 列: {localization_path}")
            return {}

    # 2. 确定是否包含 heading
    has_heading = "heading" in tmp.columns
    
    # 指定读取列
    usecols = [ts_col, "lat", "lon", "alt", "vx", "vy", "vz", "wx", "wy", "wz"]
    if has_heading:
        usecols.append("heading")
    
    try:
        df = pd.read_csv(localization_path, usecols=lambda c: c in usecols, dtype=str, on_bad_lines='skip')
    except Exception as e:
        print(f"Warning: Failed to read data of {localization_path}: {e}")
        return {}

    # 数据清洗
    def clean_numeric(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().replace('，', '').replace(',', '')
        match = re.search(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', s)
        if match: return float(match.group(0))
        return np.nan

    for col in df.columns:
        df[col] = df[col].apply(clean_numeric)

    df = df.dropna(subset=[ts_col, "lat", "lon", "alt"])
    if len(df) == 0: return {}

    # 统一时间戳
    if ts_unit == "s":
        df["timestamp_us"] = (df[ts_col] * 1e6).round().astype("int64")
    elif ts_unit == "ms":
        df["timestamp_us"] = (df[ts_col] * 1e3).round().astype("int64")
    else:
        df["timestamp_us"] = df[ts_col].round().astype("int64")
    # 去重逻辑：按时间戳去重，保留最后一条（或第一条）
    # 防止重复时间戳导致插值错误
    df = df.drop_duplicates(subset=["timestamp_us"], keep='last')
    
    # 排序：确保按时间顺序排列
    df = df.sort_values(by="timestamp_us")
     # === [新增]：强制加入滑动窗口平滑滤波 ===
    # 应对 18g 这种高频噪声，Window=10 比较合适
    try:
        window_size = 10
        print(f"  Doing applying smoothing (window={window_size})...")
        cols_to_smooth = ['lat', 'lon', 'alt']
        for col in cols_to_smooth:
            if col in df.columns:
                # min_periods=1 保证开头结尾不丢失
                # center=True 防止平滑导致相位滞后（位置偏移）
                df[col] = df[col].rolling(window=window_size, min_periods=1, center=True).mean()
    except Exception as e:
        print(f"  Warning: Smoothing failed: {e}")
    # === 项目1优化：使用 pyproj 进行 UTM 投影 ===
    # 自动计算 UTM Zone
    mean_lon = df["lon"].mean()
    mean_lat = df["lat"].mean()
    
    # 计算 UTM 区域号 (经度 + 180) / 6 + 1
    utm_zone = int((mean_lon + 180) / 6) + 1
    # 判断北半球还是南半球
    is_north = mean_lat >= 0
    
    # 构建 EPSG 代码 (北半球 326xx, 南半球 327xx)
    epsg_code = f"epsg:{32600 + utm_zone}" if is_north else f"epsg:{32700 + utm_zone}"
    
    print(f"  [Coordinates] Auto-detected UTM Zone: {utm_zone} ({'North' if is_north else 'South'}), EPSG: {epsg_code}")
    
    # 创建转换器
    transformer = Transformer.from_crs("epsg:4326", epsg_code, always_xy=True)
    
    # 批量转换 (lon, lat) -> (easting, northing)
    # 注意：pyproj always_xy=True 时输入为 (lon, lat)
    utm_x, utm_y = transformer.transform(df["lon"].values, df["lat"].values)
    
    # 以第一帧为原点，保持局部坐标数值较小，避免精度丢失
    ref_x = utm_x[0]
    ref_y = utm_y[0]
    ref_alt = df.iloc[0]["alt"]
    # 记录原点元数据
    utm_origin = {
        "x": float(ref_x),
        "y": float(ref_y),
        "z": float(ref_alt),
        "utm_zone": int(utm_zone),
        "is_north": bool(is_north),
        "epsg": epsg_code
    }
    df["x"] = utm_x - ref_x
    df["y"] = utm_y - ref_y
    df["z"] = df["alt"] - ref_alt

    # === 项目2优化：处理 Heading ===
    # 如果有 heading 列，预先计算 yaw (弧度)
    if has_heading:
        # 转换公式: yaw = radians(90 - heading)
        # heading 单位是度
        df["yaw_rad"] = np.radians(90.0 - df["heading"])
    else:
        df["yaw_rad"] = np.nan

    timestamp_to_pose = {
        int(row["timestamp_us"]): {
            "translation": [float(row["x"]), float(row["y"]), float(row["z"])],
            "velocity": [float(row.get("vx", 0.0)), float(row.get("vy", 0.0)), float(row.get("vz", 0.0))],
            "angular_velocity": [float(row.get("wx", 0.0)), float(row.get("wy", 0.0)), float(row.get("wz", 0.0))],
            "yaw": float(row["yaw_rad"]) if not pd.isna(row.get("yaw_rad")) else None # 存储 Yaw
        }
        for _, row in df.iterrows()
    }
    return timestamp_to_pose, utm_origin

def create_nuscenes_ego_poses(samples, custom_annotations, localization_dir, path_mapping=None, zero_pose=False):
    if zero_pose:
        # 使用零位姿，大幅提高处理速度
        ego_poses = []
        for i, sample in enumerate(samples):
            ego_pose = {
                "token": str(uuid.uuid4()),
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],  # 四元数表示的单位旋转
                "timestamp": sample["timestamp"],
                "next": "",
                "prev": "",
                "scene_token": sample["scene_token"]
            }
            
            if i > 0:
                ego_pose["prev"] = ego_poses[i-1]["token"]
                ego_poses[i-1]["next"] = ego_pose["token"]
            ego_poses.append(ego_pose)
            
        return ego_poses, {}            
    
    """创建NuScenes自车位姿 - 使用线性插值"""
    ego_poses = []

    # 修正 bev_date 目录提取
    localization_files = {}
    for root, _, files in os.walk(localization_dir):
        if "localzation" in Path(root).parts:
            continue
        for file in files:
            if file == "localization.csv":
                csv_path = Path(root) / file
                if csv_path.parent.name == "ok_data":
                    bev_date_dir = csv_path.parent.parent.name
                else:
                    bev_date_dir = csv_path.parent.name
                localization_files[bev_date_dir] = str(csv_path)

    if not localization_files:
        raise FileNotFoundError(f"No localization.csv files found in {localization_dir}")

    print(f"Found localization files: {list(localization_files.keys())}")

    # 加载所有定位数据
    timestamp_to_pose = {}
    map_meta_info = {} 
    for bev_dir, fp in localization_files.items():
        try:
            # 【修改】接收两个返回值
            d, origin = load_localization_data(fp)
            timestamp_to_pose.update(d)
            map_meta_info[bev_dir] = origin # 记录该 bev_date 的原点
        except Exception as e:
            print(f"Warning: failed to load {fp}: {e}")
            continue

    if not timestamp_to_pose:
        raise RuntimeError("Localization loading failed or contains no entries.")

    # 排序时间戳
    sorted_loc_timestamps = sorted(timestamp_to_pose.keys())

    # === 优化：计算或获取 Yaw 角 ===
    loc_quats = {} # ts -> [x, y, z, w]
    
    for i, t in enumerate(sorted_loc_timestamps):
        pose = timestamp_to_pose[t]
        
        # 1. 优先使用直接读取的 Heading (已转换为 Yaw)
        if pose.get("yaw") is not None:
            yaw = pose["yaw"]
        
        # 2. 如果没有 Heading (老数据)，则根据 UTM 轨迹计算
        # 由于现在使用的是 UTM 坐标，dx/dy 是真实的米制距离，计算出的角度比经纬度估算更准
        else:
            if i + 1 < len(sorted_loc_timestamps):
                t2 = sorted_loc_timestamps[i + 1]
                p1 = pose["translation"]
                p2 = timestamp_to_pose[t2]["translation"]
            else:
                # 最后一个点，参考前一个点
                t2 = t
                p1 = timestamp_to_pose[sorted_loc_timestamps[i-1]]["translation"]
                p2 = pose["translation"]
                
            dx, dy = (p2[0] - p1[0]), (p2[1] - p1[1])
            # 只有当移动距离足够大时才更新 Yaw，否则保持 0 或上一个值可能更好
            # 这里简单处理，如果静止则 yaw=0 (或者你可以优化为保持上一帧 yaw)
            yaw = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0.0
            
        # 假设 roll=0, pitch=0 (平面假设)
        r = R.from_euler('z', yaw)
        loc_quats[t] = r.as_quat()

    # 使用 bisect 进行快速查找和插值    
    # 允许的最大插值间隔（例如 0.25秒，为 250,000 微秒），超过这个间隔认为定位中断
    # 
    MAX_INTERP_GAP_US = 350_000 
    #在无法插值（如首尾或断点处）时，允许使用最近邻匹配的最大时间差
    TOLERANCE_US = 80_000  # 80ms 容忍度
    for sample in samples:
        raw_ts = sample.get("timestamp", None)
        ts_us = _normalize_to_us(raw_ts)

        # === 调试修改开始 ===
        # 尝试增加或减少 100ms (100000us)，观察旋转是否变小
        # 如果顺时针旋转问题改善，说明原本的 Pose 滞后了
        # TIME_OFFSET_US = 70000  # 试着加 50ms 或 100ms
        # ts_us = ts_us + TIME_OFFSET_US 
        # === 调试修改结束 ===
        if ts_us is None:
            continue

        # 找到插入位置
        idx = bisect.bisect_right(sorted_loc_timestamps, ts_us)
        
        # 情况1：时间戳在所有定位数据之前
        if idx == 0:
            closest_ts = sorted_loc_timestamps[0]
            if abs(closest_ts - ts_us) < TOLERANCE_US: # 80ms 容忍
                pose_data = timestamp_to_pose[closest_ts]
                rotation = loc_quats[closest_ts]
            else:
                print(f"Warning: sample ts {ts_us} is too early (first loc {closest_ts}), skip")
                continue

        # 情况2：时间戳在所有定位数据之后
        elif idx == len(sorted_loc_timestamps):
            closest_ts = sorted_loc_timestamps[-1]
            if abs(closest_ts - ts_us) < TOLERANCE_US:
                pose_data = timestamp_to_pose[closest_ts]
                rotation = loc_quats[closest_ts]
            else:
                print(f"Warning: sample ts {ts_us} is too late (last loc {closest_ts}), skip")
                continue

        # 情况3：在两个定位点之间 -> 插值
        else:
            t0 = sorted_loc_timestamps[idx - 1]
            t1 = sorted_loc_timestamps[idx]
            
            # 检查间隔是否过大
            if (t1 - t0) > MAX_INTERP_GAP_US:
                # 间隔太大，退化为最近邻
                if abs(ts_us - t0) < abs(ts_us - t1):
                    if abs(ts_us - t0) < TOLERANCE_US:
                        pose_data = timestamp_to_pose[t0]
                        rotation = loc_quats[t0]
                    else:
                        print(f"Warning: sample ts {ts_us} in large gap, too far from t0")
                        continue
                else:
                    if abs(ts_us - t1) < TOLERANCE_US:
                        pose_data = timestamp_to_pose[t1]
                        rotation = loc_quats[t1]
                    else:
                        print(f"Warning: sample ts {ts_us} in large gap, too far from t1")
                        continue
            else:
                # 执行线性插值
                ratio = (ts_us - t0) / (t1 - t0)
                
                p0 = timestamp_to_pose[t0]
                p1 = timestamp_to_pose[t1]
                
                # 插值平移
                trans = [
                    p0["translation"][0] + (p1["translation"][0] - p0["translation"][0]) * ratio,
                    p0["translation"][1] + (p1["translation"][1] - p0["translation"][1]) * ratio,
                    p0["translation"][2] + (p1["translation"][2] - p0["translation"][2]) * ratio
                ]
                
                # 插值速度
                vel = [
                    p0["velocity"][0] + (p1["velocity"][0] - p0["velocity"][0]) * ratio,
                    p0["velocity"][1] + (p1["velocity"][1] - p0["velocity"][1]) * ratio,
                    p0["velocity"][2] + (p1["velocity"][2] - p0["velocity"][2]) * ratio
                ]
                
                # 插值角速度
                ang_vel = [
                    p0["angular_velocity"][0] + (p1["angular_velocity"][0] - p0["angular_velocity"][0]) * ratio,
                    p0["angular_velocity"][1] + (p1["angular_velocity"][1] - p0["angular_velocity"][1]) * ratio,
                    p0["angular_velocity"][2] + (p1["angular_velocity"][2] - p0["angular_velocity"][2]) * ratio
                ]
                
                # 插值旋转 (Slerp)
                q0 = loc_quats[t0]
                q1 = loc_quats[t1]
                
                # 手动 Slerp 或者使用 scipy
                # 这里使用简单的近似：如果角度变化小，线性插值归一化也行，但 Slerp 更好
                # 为简单起见，这里使用 scipy 的 Slerp
                times = [0, 1]
                key_rots = R.from_quat([q0, q1])
                slerp_func = Slerp(times, key_rots)
                interp_rot = slerp_func([ratio])
                q_interp = interp_rot.as_quat()[0] # [x, y, z, w]
                
                # 构造结果
                pose_data = {
                    "translation": trans,
                    "velocity": vel,
                    "angular_velocity": ang_vel
                }
                rotation = q_interp

        # 构造最终的 ego_pose
        # 注意：NuScenes rotation 顺序是 [w, x, y, z]
        final_rotation = [float(rotation[3]), float(rotation[0]), float(rotation[1]), float(rotation[2])]
        
        ego_pose = {
            "token": str(uuid.uuid4()),
            "timestamp": int(ts_us), # 使用样本的时间戳
            "rotation": final_rotation,
            "translation": pose_data["translation"],
            "velocity": pose_data.get("velocity", [0.0, 0.0, 0.0]),
            "angular_velocity": pose_data.get("angular_velocity", [0.0, 0.0, 0.0])
        }
        ego_poses.append(ego_pose)

    return ego_poses, map_meta_info

def debug_coordinate_system(box_3d, extrinsic_matrix):
    """调试坐标系变换"""
    # 原始3D框参数
    translation = [box_3d[0], box_3d[1], box_3d[2]]
    size = [box_3d[6], box_3d[7], box_3d[8]]  # length, width, height
    
    print("=== 坐标系调试 ===")
    print(f"原始3D框 - 位置: {translation}")
    print(f"原始3D框 - 尺寸: {size} (length, width, height)")
    
    # 测试不同的尺寸顺序
    size_combinations = {
        "length, width, height": [size[0], size[1], size[2]],
        "width, length, height": [size[1], size[0], size[2]], 
        "height, width, length": [size[2], size[1], size[0]],
        "width, height, length": [size[1], size[2], size[0]]
    }
    
    for name, sizes in size_combinations.items():
        print(f"尺寸顺序 '{name}': {sizes}")
    
    return size_combinations

def quat_to_pitch(w, x, y, z):
    """四元数 (w,x,y,z) → pitch 弧度"""
    sinp = 2 * (w * y - z * x)
    cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(sinp, cosp)

def quat_to_roll(w, x, y, z):
    """从四元数 (w,x,y,z) 解算 roll 角（弧度）"""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    return np.arctan2(sinr_cosp, cosr_cosp)

# ---------- 1. 使用Numba加速点云计数 ----------
# Numba加速的点云计数核心函数
@nb.njit(fastmath=True, cache=True)
def _points_in_bbox_numba(points, center, half_size, rot_matrix):
    """Numba加速的边界框内点计数核心函数"""
    n = points.shape[0]
    count = 0
    for i in range(n):
        # 平移
        x = points[i, 0] - center[0]
        y = points[i, 1] - center[1]
        z = points[i, 2] - center[2]
        # 旋转到局部坐标系
        xl = rot_matrix[0,0]*x + rot_matrix[1,0]*y + rot_matrix[2,0]*z
        yl = rot_matrix[0,1]*x + rot_matrix[1,1]*y + rot_matrix[2,1]*z
        zl = rot_matrix[0,2]*x + rot_matrix[1,2]*y + rot_matrix[2,2]*z
        # 判断是否在边界框内
        if (abs(xl) <= half_size[0] and 
            abs(yl) <= half_size[1] and 
            abs(zl) <= half_size[2]):
            count += 1
    return count

def count_points_in_bbox_numba(points, translation, size, rotation):
    """
    使用Numba加速的3D边界框点云计数
    """
    if points is None or len(points) == 0:
        return 0
    
    try:
        # 确保输入数据格式正确
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)
        
        # 确保points是(N, 3)形状
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        elif points.shape[1] > 3:
            points = points[:, :3]
        
        # 计算旋转矩阵
        rot_obj = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])
        rotation_matrix = rot_obj.as_matrix().astype(np.float32)
        
        # 准备Numba函数参数
        center = np.array(translation, dtype=np.float32)
        half_size = np.array(size, dtype=np.float32) / 2.0
        
        # 调用Numba加速函数
        count = _points_in_bbox_numba(
            points.astype(np.float32),
            center,
            half_size,
            rotation_matrix
        )
        
        # 调试输出 - 对于点数异常的情况
        # if count < 3:
        #     print(f"调试 - 边界框内点数: {count}")
        #     print(f"  边界框中心: {center}")
        #     print(f"  边界框尺寸: {size}")
        #     print(f"  点云总数: {len(points)}")
        
        return int(count)
        
    except Exception as e:
        print(f"计算边界框内点数时出错: {e}")
        import traceback
        traceback.print_exc()
        return 10

# 在 convert_to_nuscenes_samples_optimized.py 中添加

@nb.njit(fastmath=True, parallel=True, cache=True)
def _points_in_bbox_batch_numba(points, centers, half_sizes, rot_matrices):
    """批量处理多个边界框的点云计数"""
    n_boxes = centers.shape[0]
    n_points = points.shape[0]
    counts = np.zeros(n_boxes, dtype=np.int32)
    
    for i in nb.prange(n_boxes):
        count = 0
        center = centers[i]
        half_size = half_sizes[i]
        rot_matrix = rot_matrices[i]
        
        for j in range(n_points):
            # 平移
            x = points[j, 0] - center[0]
            y = points[j, 1] - center[1]
            z = points[j, 2] - center[2]
            # 旋转到局部坐标系
            xl = rot_matrix[0,0]*x + rot_matrix[1,0]*y + rot_matrix[2,0]*z
            yl = rot_matrix[0,1]*x + rot_matrix[1,1]*y + rot_matrix[2,1]*z
            zl = rot_matrix[0,2]*x + rot_matrix[1,2]*y + rot_matrix[2,2]*z
            # 判断是否在边界框内
            if (abs(xl) <= half_size[0] and 
                abs(yl) <= half_size[1] and 
                abs(zl) <= half_size[2]):
                count += 1
        counts[i] = count
    return counts

def count_points_in_bboxes_batch(points, translations, sizes, rotations):
    """批量计算多个边界框内的点数"""
    if points is None or len(points) == 0:
        return [10] * len(translations)
    
    try:
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)
        
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        elif points.shape[1] > 3:
            points = points[:, :3]
        
        n_boxes = len(translations)
        centers = np.array(translations, dtype=np.float32)
        half_sizes = np.array(sizes, dtype=np.float32) / 2.0
        
        # 预计算所有旋转矩阵
        rot_matrices = np.zeros((n_boxes, 3, 3), dtype=np.float32)
        for i in range(n_boxes):
            rot_obj = R.from_quat([rotations[i][1], rotations[i][2], rotations[i][3], rotations[i][0]])
            rot_matrices[i] = rot_obj.as_matrix().astype(np.float32)
        
        counts = _points_in_bbox_batch_numba(
            points.astype(np.float32),
            centers,
            half_sizes,
            rot_matrices
        )
        
        return counts.tolist()
        
    except Exception as e:
        print(f"批量计算边界框内点数时出错: {e}")
        return [10] * len(translations)
# 可选：也提供一个不使用Numba的版本作为备选
def count_points_in_bbox_accurate(points, translation, size, rotation):
    """
    不使用Numba的准确计数函数（兼容性备选）
    """
    if points is None or len(points) == 0:
        return 0
    
    try:
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)
        
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        elif points.shape[1] > 3:
            points = points[:, :3]
        
        center = np.array(translation, dtype=np.float32)
        half_size = np.array(size, dtype=np.float32) / 2.0
        
        rot_obj = R.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])
        rotation_matrix = rot_obj.as_matrix().astype(np.float32)
        
        points_centered = points - center
        points_local = points_centered @ rotation_matrix.T
        
        in_bbox_mask = (
            (np.abs(points_local[:, 0]) <= half_size[0]) &
            (np.abs(points_local[:, 1]) <= half_size[1]) &
            (np.abs(points_local[:, 2]) <= half_size[2])
        )
        
        count = np.sum(in_bbox_mask)
        
        # if count < 3:
        #     print(f"调试 - 边界框内点数: {count}")
        #     print(f"  边界框中心: {center}")
        #     print(f"  边界框尺寸: {size}")
        #     print(f"  点云总数: {len(points)}")
        
        return int(count)
        
    except Exception as e:
        print(f"计算边界框内点数时出错: {e}")
        return 10

def create_nuscenes_sample_annotations(custom_annotations, samples, instances,
        instance_key_to_token, custom_to_nuscenes, categories, attributes,
        visibility, path_mapping, seq_data, ego_poses, calibrated_sensors, sensors,
        localization_dir,use_numba=True):
    """创建NuScenes样本标注 - 将激光系框转换到全局系"""
    sample_annotations = []
    seq_data_root = Path(seq_data).parent if seq_data else None
    # attribute_name_to_token = {attr["name"]: attr["token"] for attr in attributes}

    # 【优化】构建有序的时间戳列表和对应的 ego_pose 列表
    # 假设 ego_poses 已经按时间排序（通常是，但为了保险可以再排一次）
    sorted_ego_poses = sorted(ego_poses, key=lambda x: x['timestamp'])
    ego_timestamps = [ep['timestamp'] for ep in sorted_ego_poses]
    # LIDAR_TOP 外参
    lidar_calib = None
    for cs in calibrated_sensors:
        for s in sensors:
            if s["token"] == cs["sensor_token"] and s["channel"] == "LIDAR_TOP":
                lidar_calib = cs
                break

    # 辅助：根据样本时间戳找到最近 ego_pose, 使用二分查找替代线性遍历
    def _ego_for_sample(sample_ts_us, max_diff=50_000):
        # 使用 bisect_left 找到插入位置
        idx = bisect.bisect_left(ego_timestamps, sample_ts_us)
        
        best_ep = None
        min_diff = float('inf')
        
        # 检查 idx 位置（大于等于 sample_ts_us 的第一个）
        if idx < len(ego_timestamps):
            diff = abs(ego_timestamps[idx] - sample_ts_us)
            if diff < min_diff:
                min_diff = diff
                best_ep = sorted_ego_poses[idx]
        
        # 检查 idx-1 位置（小于 sample_ts_us 的最后一个）
        if idx > 0:
            diff = abs(ego_timestamps[idx-1] - sample_ts_us)
            if diff < min_diff:
                min_diff = diff
                best_ep = sorted_ego_poses[idx-1]
                
        if min_diff <= max_diff:
            return best_ep
        return None

    # 预排序样本
    samples_with_timestamps = sorted(((s["timestamp"], s) for s in samples), key=lambda x: x[0])

    # 点云缓存
    point_cloud_cache = {}
    instance_annotations = defaultdict(list)

    count_function = count_points_in_bbox_numba if use_numba else count_points_in_bbox_accurate
    if use_numba:
        print("使用Numba加速的点云计数")
    else:
        print("使用纯Python点云计数")
    #冗余：ego_poses 已经在 main 函数中生成并作为参数传进来了。这里再次生成非常浪费时间
    #暂留注释以防万一
    # try:
    #     real_ego_poses_list, _ = create_nuscenes_ego_poses(
    #         samples, custom_annotations, localization_dir, path_mapping, zero_pose=False
    #     )
    #     print("成功加载真实位姿用于速度计算。")
    # except Exception as e:
    #     print(f"警告: 无法加载真实位姿 ({e})，标注中的速度将默认为 0。")
    #     # 如果失败，回退到零位姿，这样后续代码不会报错，但速度计算结果为0
    #     real_ego_poses_list, _ = create_nuscenes_ego_poses(
    #         samples, custom_annotations, localization_dir, path_mapping, zero_pose=True
    #     )
    # # 构建查找表
    # ts_to_real_pose = {ep['timestamp']: ep for ep in real_ego_poses_list}
    for timestamp, sample in samples_with_timestamps:
        # 找匹配帧
        matching_frame, closest_diff, matching_timestamp, matching_frame_idx = None, float("inf"), None, None
        for frame_idx, frame in enumerate(custom_annotations):
            info_url = frame.get("info", "")
            frame_ts = _normalize_to_us(Path(info_url).stem) if info_url else None
            time_diff = abs(frame_ts - timestamp) if frame_ts is not None else 1e18
            if time_diff < closest_diff and time_diff < 200_000:  # 0.2s
                closest_diff, matching_frame = time_diff, frame
                matching_timestamp, matching_frame_idx = frame_ts, frame_idx

        if matching_frame is None:
            print(f"警告: 未找到时间戳 {timestamp} 的匹配帧")
            continue

        sequence_id = matching_frame.get("_id", "sequence_default")

        # 点云加载
        point_cloud = None
        cache_key = f"{sequence_id}_{matching_timestamp}"
        if cache_key in point_cloud_cache:
            point_cloud = point_cloud_cache[cache_key]
        else:
            try:
                pcd_info = Path(matching_frame.get("info", ""))
                bin_pth = seq_data_root / Path(path_mapping[pcd_info.parent.parent.name]) / Path(
                    pcd_info.parent.name) / (pcd_info.stem + ".bin")
                if bin_pth.exists():
                    pc = np.fromfile(bin_pth, dtype=np.float32).reshape(-1, 5)[:, :3]
                    point_cloud_cache[cache_key] = pc
                    point_cloud = pc
                else:
                    print(f"警告: 点云文件不存在: {bin_pth}")
            except Exception as e:
                print(f"警告: 读取点云文件失败: {e}")

        # 当前样本的 ego_pose
        ego_pose = _ego_for_sample(timestamp)
        if ego_pose is None:
            print(f"警告: 样本 {sample['token']} 未匹配到 ego_pose，跳过标注")
            continue

        # 预备旋转矩阵
        R_ego = R.from_quat([ego_pose["rotation"][1], ego_pose["rotation"][2],
                             ego_pose["rotation"][3], ego_pose["rotation"][0]]).as_matrix()
        t_ego = np.array(ego_pose["translation"])
        if lidar_calib:
            R_lidar = R.from_quat([lidar_calib["rotation"][1], lidar_calib["rotation"][2],
                                   lidar_calib["rotation"][3], lidar_calib["rotation"][0]]).as_matrix()
            t_lidar = np.array(lidar_calib["translation"])
        else:
            R_lidar = np.eye(3)
            t_lidar = np.zeros(3)

        for label in matching_frame.get("labels", []):
            points = label.get("points", [])
            if len(points) != 9:
                continue
            tx, ty, tz, roll, pitch, yaw, length, width, height = points
            # 1. 原始标注是在 LiDAR 坐标系下的 (假设标注工具输出的就是 LiDAR 系)
            # 构造 LiDAR 系下的旋转矩阵
            r_box_lidar = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
            t_box_lidar = np.array([tx, ty, tz])
            size = [width, length, height]   # NuScenes: l,w,h

            # 标注就是在点云上标的，那它就是相对于 LiDAR 原点的。
            # 此时 t_lidar 和 R_lidar 应该是用来把 LiDAR 系转到 Ego 系的。
            
            # 正确的计算：
            # Point_Lidar = [x, y, z]
            # Point_Ego = R_lidar @ Point_Lidar + t_lidar
            # Point_Global = R_ego @ Point_Ego + t_ego
            
            # 所以，如果 t_box 是 LiDAR 系下的中心点：
            t_box_ego = t_lidar + R_lidar @ t_box_lidar
            t_glob = t_ego + R_ego @ t_box_ego
            
            R_glob_mat = R_ego @ R_lidar @ r_box_lidar
            q_glob = R.from_matrix(R_glob_mat).as_quat()
            
            rotation_global = [float(q_glob[3]), float(q_glob[0]), float(q_glob[1]), float(q_glob[2])]
            translation_global = t_glob.tolist()

            label_id = label.get("id")
            instance_key = f"{sequence_id}_{label_id}"
            instance_token = instance_key_to_token.get(instance_key, "")
            if not instance_token:
                continue
            # 关于类别映射，如需调试，请取消注释
            # label_name = label.get("label", "").lower()
            # nuscenes_category = custom_to_nuscenes.get(label_name)
            # if nuscenes_category is None:
            #     eng = extract_english_category(label_name)
            #     fallback = {
            #         "bus": "vehicle.bus.rigid",
            #         "truck": "vehicle.truck",
            #         "motorcycle": "vehicle.motorcycle",
            #         "bicycle": "vehicle.bicycle",
            #         "pedestrian": "human.pedestrian.adult",
            #         "animal": "animal",
            #         "barrier": "movable_object.barrier",
            #         "cone": "movable_object.trafficcone",
            #         "trailer": "vehicle.trailer",
            #         "construction": "vehicle.construction",
            #         "car": "vehicle.car",
            #     }
            #     nuscenes_category = fallback.get(eng, "vehicle.car")
            #     custom_to_nuscenes[label_name] = nuscenes_category

            # 计点
            # 必须使用 LiDAR 坐标系下的框参数
            if point_cloud is not None and len(point_cloud) > 0:
                try:
                    # 构造 LiDAR 系下的四元数 [w, x, y, z] 用于传入 count_function
                    # count_function 内部会用它把点云转到框的局部坐标系
                    q_lidar = R.from_matrix(r_box_lidar).as_quat()
                    rotation_lidar_fmt = [float(q_lidar[3]), float(q_lidar[0]), float(q_lidar[1]), float(q_lidar[2])]
                    
                    # 【核心修复】传入 LiDAR 系下的位姿
                    num_lidar_pts = count_function(point_cloud, t_box_lidar, size, rotation_lidar_fmt)
                except Exception:
                    num_lidar_pts = 0
            else:
                num_lidar_pts = 0
            num_radar_pts = 0
            # ===  velocity 字段 如需计算，代码如下 ===
            # vel_xy = calculate_global_velocity(
            #     custom_annotations, 
            #     sequence_id, 
            #     label_id, 
            #     matching_frame_idx,
            #     ts_to_real_pose 
            # )
            # velocity = [float(vel_xy[0]), float(vel_xy[1])]

            # 属性/可见性
            attr_tokens = []
            visibility_token = "4"
            attr = label.get("attr", {})
            if "遮挡属性" in attr:
                if "有遮挡" in attr["遮挡属性"] or "完全遮挡" in attr["遮挡属性"]:
                    visibility_token = "1"
                elif "大部分遮挡" in attr["遮挡属性"]:
                    visibility_token = "2"
                elif "部分遮挡" in attr["遮挡属性"]:
                    visibility_token = "3"
                else:
                    visibility_token = "4"

            sample_annotation = {
                "token": str(uuid.uuid4()),
                "sample_token": sample["token"],
                "instance_token": instance_token,
                "visibility_token": visibility_token,
                "attribute_tokens": attr_tokens,
                # "category_name": nuscenes_category,
                "translation": translation_global,
                "size": size,
                "rotation": rotation_global,
                "num_lidar_pts": num_lidar_pts,
                "num_radar_pts": num_radar_pts,
                # "velocity": velocity,  #  velocity 字段，单位 m/s
                "prev": "",
                "next": ""
            }

            sample_annotations.append(sample_annotation)
            instance_annotations[instance_token].append(sample_annotation)
            sample.setdefault("anns", []).append(sample_annotation["token"])

    # 同一实例内 prev/next
    print("正在构建实例链表并检查速度异常...")
    # 构建 sample_token -> timestamp 的快速查找表 (优化性能)
    sample_token_to_ts = {s["token"]: s["timestamp"] for _, s in samples_with_timestamps}
    
    # 速度阈值：50 m/s (180 kph)。超过此速度视为异常跳变，断开连接。
    VELOCITY_THRESHOLD = 50.0 
    cut_count = 0

    for instance in instances:
        anns = instance_annotations.get(instance["token"], [])
        if not anns:
            continue
            
        # 【优化 2】使用查找表进行排序，比原来的 lambda 快很多
        anns_sorted = sorted(anns, key=lambda a: sample_token_to_ts.get(a["sample_token"], 0))
        
        instance["nbr_annotations"] = len(anns_sorted)
        instance["first_annotation_token"] = anns_sorted[0]["token"]
        instance["last_annotation_token"] = anns_sorted[-1]["token"]
        
        for i, ann in enumerate(anns_sorted):
            # 默认不连接，只有检查通过才连接
            # ann["prev"] = ""  (已在初始化时默认为空)
            # ann["next"] = ""
            
            if i > 0:
                prev_ann = anns_sorted[i-1]
                
                # --- 【新增 3】速度熔断检查 ---
                
                # 1. 获取时间差 (秒)
                t1 = sample_token_to_ts.get(prev_ann["sample_token"], 0)
                t2 = sample_token_to_ts.get(ann["sample_token"], 0)
                dt = (t2 - t1) / 1e6
                
                # 2. 获取空间距离 (米)
                p1 = np.array(prev_ann["translation"])
                p2 = np.array(ann["translation"])
                dist = np.linalg.norm(p2 - p1)
                
                # 3. 计算速度
                is_valid_link = True
                if dt > 0.0001: # 防止除以零
                    v = dist / dt
                    if v > VELOCITY_THRESHOLD:
                        is_valid_link = False
                        cut_count += 1
                        # 可选：打印警告（如果太多可以注释掉）
                        if cut_count <= 10:
                            print(f"警告: 实例 {instance['token']} 发生跳变 (v={v:.1f} m/s), 断开连接。")
                
                # 4. 只有速度正常才连接
                if is_valid_link:
                    ann["prev"] = prev_ann["token"]
                    prev_ann["next"] = ann["token"]

    print(f"=== 样本标注处理完成 ===")
    print(f"总共处理了 {len(sample_annotations)} 个标注")
    print(f"因速度异常断开了 {cut_count} 处实例连接")
    return sample_annotations

def create_nuscenes_map():
    """创建NuScenes地图"""
    map = {
        "token": str(uuid.uuid4()),
        "log_tokens": [],  # 稍后填充
        "category": "semantic_prior",
        "filename": "custom_map.json"
    }

    return map
