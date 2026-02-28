import os
import json, uuid
# import shutil
from collections import defaultdict
import argparse
import numpy as np

class NuScenesMerger:
    def __init__(self):
        self.token_mapping = defaultdict(dict)
        self.dataset_idx_map = {}  # 记录每个token来自哪个数据集
        
    def generate_new_token(self, old_token, token_type):
        """生成新的唯一token，保持UUID格式"""
        if not old_token:  # 处理空token
            return str(uuid.uuid4())
    
        # 【优化点：优先检查是否已有映射】
        if old_token in self.token_mapping[token_type]:
            return self.token_mapping[token_type][old_token]
        
        # 如果没有映射，则生成新的
        new_token = str(uuid.uuid4())
        self.token_mapping[token_type][old_token] = new_token
        return new_token
    
    def _populate_annotation_prev_next_by_spatial_matching(self, merged_data, max_distance=2.0):
        """✅ 改进版：基于空间匹配填充 prev/next，同时尊重 instance 分组"""
        print(f"使用空间匹配填充 sample_annotation.prev/next（备选方案），阈值={max_distance}m ...")
        
        # 构建 sample token -> sample 的映射
        sample_by_token = {s['token']: s for s in merged_data['sample']}
        
        # 构建 sample.next 映射
        sample_next = {s['token']: s.get('next', '') for s in merged_data['sample']}
        
        # 按 instance 分组进行空间匹配
        annotations_by_instance = {}
        for ann in merged_data['sample_annotation']:
            inst_token = ann.get('instance_token', '')
            if inst_token:
                if inst_token not in annotations_by_instance:
                    annotations_by_instance[inst_token] = []
                annotations_by_instance[inst_token].append(ann)
        
        filled = 0
        
        # 对每个 instance 分别进行空间匹配
        for inst_token, anns in annotations_by_instance.items():
            # 按 sample 时间戳排序
            anns_sorted = []
            for ann in anns:
                sample = sample_by_token.get(ann.get('sample_token', ''))
                if sample:
                    anns_sorted.append((sample.get('timestamp', 0), ann))
            
            anns_sorted.sort(key=lambda x: x[0])
            
            # 对相邻的两个 annotation 进行空间匹配
            for i in range(len(anns_sorted) - 1):
                src_ann = anns_sorted[i][1]
                dst_ann = anns_sorted[i+1][1]
                
                # 计算两个框中心的距离
                src_pos = np.array(src_ann.get('translation', [0, 0, 0])[:2], dtype=np.float32)
                dst_pos = np.array(dst_ann.get('translation', [0, 0, 0])[:2], dtype=np.float32)
                distance = np.linalg.norm(dst_pos - src_pos)
                
                # 如果距离在阈值内，建立链接
                if distance <= max_distance:
                    if src_ann.get('next') != dst_ann['token']:
                        src_ann['next'] = dst_ann['token']
                        filled += 1
                    if dst_ann.get('prev') != src_ann['token']:
                        dst_ann['prev'] = src_ann['token']
                        filled += 1
        
        print(f"  空间匹配完成，填充/更新 {filled} 个 prev/next 字段")
        return filled
    def merge_datasets(self, dataset_folders, output_folder):
        """合并多个nuScenes数据集"""
        
        os.makedirs(output_folder, exist_ok=True)
        
        # 初始化合并后的数据结构
        merged_data = {
            'category': [], 'attribute': [], 'visibility': [], 'instance': [],
            'sensor': [], 'calibrated_sensor': [], 'ego_pose': [], 'log': [],
            'scene': [], 'sample': [], 'sample_data': [], 'sample_annotation': [], 'map': []
        }
        
        # 第一阶段：收集所有数据，不进行任何修改
        all_raw_data = {key: [] for key in merged_data.keys()}
        
        for dataset_idx, dataset_folder in enumerate(dataset_folders):
            print(f"收集数据集 {dataset_idx + 1}: {dataset_folder}")
            
            for data_type in merged_data.keys():
                json_file = os.path.join(dataset_folder, f"{data_type}.json")
                
                if not os.path.exists(json_file):
                    print(f"警告: 在 {dataset_folder} 中未找到 {data_type}.json")
                    continue
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                all_raw_data[data_type].extend(data)
        
        # 第二阶段：合并基本数据（无依赖关系）
        self._merge_independent_data(all_raw_data, merged_data)
        
        # 第三阶段：合并有依赖关系的数据
        self._merge_dependent_data(all_raw_data, merged_data)
        
        # 第四阶段：修复所有引用关系
        self._fix_all_references(merged_data)
        #  生成脚本已经处理了速度熔断，这里再次填充会破坏熔断逻辑，导致速度再次爆表。故注释掉一下一行。
        # 使用 sample 的 prev/next 填充 sample_annotation 的 prev/next（快速、基于索引）
        # self._populate_annotation_prev_next_using_samples(merged_data)
        
        # 如果基于instance_token的匹配失败，使用空间匹配作为fallback

        # filled_spatial = self._populate_annotation_prev_next_by_spatial_matching(merged_data, max_distance=2.0)
        # if filled_spatial > 0:
        #     print(f"  使用空间匹配补充了 {filled_spatial} 个 prev/next")
        # annotations_with_next = sum(1 for ann in merged_data['sample_annotation'] 
        #                            if ann.get('next', ''))
        # if annotations_with_next < len(merged_data['sample_annotation']) * 0.1:  # 如果少于10%有next
        #     print(f"基于instance_token的匹配效果不佳（{annotations_with_next}/{len(merged_data['sample_annotation'])}），尝试空间匹配...")
        #     filled_spatial = self._populate_annotation_prev_next_by_spatial_matching(merged_data, max_distance=2.0)
        #     print(f"  空间匹配补充了 {filled_spatial} 个 prev/next 链接")
        # 第五阶段：验证数据完整性
        self._validate_data_integrity(merged_data)
        
        # 保存合并后的数据
        self._save_merged_data(merged_data, output_folder)
        
        print(f"数据集合并完成！输出位置: {output_folder}")

    def _populate_annotation_prev_next_using_samples(self, merged_data):
        """✅ 改进版：用已存在的 sample.prev/next 和 instance 分组填充 annotation.prev/next"""
        print("使用 sample.prev/next + instance 分组填充 sample_annotation.prev/next ...")
        
        # 调试信息1：统计基本信息
        total_annotations = len(merged_data['sample_annotation'])
        non_empty_instance_count = sum(1 for ann in merged_data['sample_annotation'] 
                                        if ann.get('instance_token', ''))
        
        print(f"  调试1: sample_annotation 总数={total_annotations}, 非空 instance_token={non_empty_instance_count}")
        
        # 构建 sample 的前后映射
        sample_prev = {s['token']: s.get('prev', '') for s in merged_data['sample']}
        sample_next = {s['token']: s.get('next', '') for s in merged_data['sample']}
        
        # 调试信息2：统计 sample 的 prev/next 情况
        sample_with_prev = sum(1 for p in sample_prev.values() if p)
        sample_with_next = sum(1 for n in sample_next.values() if n)
        print(f"  调试2: sample 总数={len(sample_prev)}, 有prev的={sample_with_prev}, 有next的={sample_with_next}")
        
        # ✅ 关键修复：按 instance 分组，而非全局时间序列
        annotations_by_instance = {}
        for ann in merged_data['sample_annotation']:
            inst_token = ann.get('instance_token', '')
            if inst_token:  # 只处理有 instance_token 的
                if inst_token not in annotations_by_instance:
                    annotations_by_instance[inst_token] = []
                annotations_by_instance[inst_token].append(ann)
        
        print(f"  调试3: 共 {len(annotations_by_instance)} 个实例有标注")
        
        # 构建 sample_token 到 annotation 的映射（每个 instance 内）
        ann_index = {}
        for inst_token, anns in annotations_by_instance.items():
            for ann in anns:
                key = (inst_token, ann.get('sample_token', ''))
                ann_index[key] = ann
        
        print(f"  调试4: 预构建的 ann_index 大小={len(ann_index)}")
        
        # ✅ 关键：按照每个 instance 内的 sample 时间序列填充 prev/next
        filled = 0
        updated = 0
        skipped_no_instance = 0
        skipped_no_next_sample = 0
        skipped_no_match = 0
        
        for inst_token, anns in annotations_by_instance.items():
            # 为该实例的所有 annotation 按 sample 时间戳排序
            anns_with_samples = []
            for ann in anns:
                sample_token = ann.get('sample_token', '')
                # 查找对应的 sample 时间戳
                sample_ts = 0
                for sample in merged_data['sample']:
                    if sample['token'] == sample_token:
                        sample_ts = sample.get('timestamp', 0)
                        break
                anns_with_samples.append((sample_ts, ann))
            
            # 按时间戳排序
            anns_with_samples.sort(key=lambda x: x[0])
            
            # 为同一实例的相邻 annotation 设置 prev/next
            for i, (_, ann) in enumerate(anns_with_samples):
                if i > 0:
                    prev_ann = anns_with_samples[i-1][1]
                    if ann.get('prev') != prev_ann['token']:
                        ann['prev'] = prev_ann['token']
                        updated += 1
                
                if i < len(anns_with_samples) - 1:
                    next_ann = anns_with_samples[i+1][1]
                    if ann.get('next') != next_ann['token']:
                        ann['next'] = next_ann['token']
                        filled += 1
        
        # 调试信息5：填充结果统计
        print(f"  调试5: 填充结果统计:")
        print(f"    成功填充 next 字段: {filled} 个")
        print(f"    成功更新 prev 字段: {updated} 个")
        print(f"    总共处理: {filled + updated} 个 annotation")
        
        # 调试信息6：检查填充后的结果
        annotations_with_prev = sum(1 for ann in merged_data['sample_annotation'] if ann.get('prev', ''))
        annotations_with_next = sum(1 for ann in merged_data['sample_annotation'] if ann.get('next', ''))
        print(f"  调试6: 填充后统计 - 有prev的标注: {annotations_with_prev}, 有next的标注: {annotations_with_next}")
        
        # 调试信息7：输出一个具体的例子用于调试
        print("  调试7: 具体例子 - 检查第一个有 instance_token 的 annotation:")
        for ann in merged_data['sample_annotation']:
            if ann.get('instance_token', ''):
                print(f"    token: {ann.get('token', '')[:16]}...")
                print(f"    instance_token: {ann.get('instance_token', '')[:16]}...")
                print(f"    sample_token: {ann.get('sample_token', '')[:16]}...")
                print(f"    prev: {ann.get('prev', '')[:16] if ann.get('prev') else '空'}")
                print(f"    next: {ann.get('next', '')[:16] if ann.get('next') else '空'}")
                break
        
        print(f"  总计: 填充/更新 {filled + updated} 个 sample_annotation.prev/next 字段")
    
    def _merge_independent_data(self, all_raw_data, merged_data):
        """合并独立数据（无外部引用）"""
        print("合并独立数据...")
        
        # 【关键修复】先合并所有category并记录映射，后续instance依赖这个映射
        # 首先创建unknown类别
        unknown_category = {
            "token": str(uuid.uuid4()),
            "name": "unknown",
            "description": "Unknown category for instances with missing category information"
        }
        merged_data['category'].append(unknown_category)
        
        # 【关键修复】合并category时必须记录token映射
        # 使用name+description作为唯一标识来去重（同名不同desc的category视为不同）
        print("  合并category...")
        seen_categories = {'unknown'}  # 已经添加了unknown
        category_name_to_new_token = {}  # 记录category name到新token的映射
        
        for item in all_raw_data['category']:
            old_token = item.get('token', '')
            name = item.get('name', '')
            
            # 如果此名称的category已存在，则复用新token
            if name in seen_categories:
                # 获取现有的新token并记录映射
                if name in category_name_to_new_token:
                    new_token = category_name_to_new_token[name]
                else:
                    # 从已合并的category中找到新token
                    for cat in merged_data['category']:
                        if cat['name'] == name:
                            new_token = cat['token']
                            category_name_to_new_token[name] = new_token
                            break
            else:
                # 生成新token并保存
                new_token = str(uuid.uuid4())
                item['token'] = new_token
                merged_data['category'].append(item)
                seen_categories.add(name)
                category_name_to_new_token[name] = new_token
            
            # 记录映射：旧token -> 新token
            self.token_mapping['category'][old_token] = new_token
        
        print(f"  category: {len(merged_data['category'])} 项 (映射 {len(self.token_mapping['category'])} 个token)")
        
        # 【修复】attribute和visibility去重
        for data_type in ['attribute', 'visibility']:
            seen_items = set()
            for item in all_raw_data[data_type]:
                identifier = item.get('name', str(item))
                if identifier not in seen_items:
                    old_token = item.get('token', '')
                    new_token = self.generate_new_token(old_token, data_type)
                    item['token'] = new_token
                    merged_data[data_type].append(item)
                    seen_items.add(identifier)
            
            print(f"  {data_type}: {len(merged_data[data_type])} 项")
        
        # 【修复】sensor不进行去重，保留每个数据集的标定参数
        print("  合并sensor（按 channel 去重）...")
        seen_sensors = {} # channel -> new_token
        
        # 1. 先把已合并数据中现有的 sensor 加入索引 (防止多次调用出错)
        for s in merged_data['sensor']:
            seen_sensors[s['channel']] = s['token']
        
        for item in all_raw_data['sensor']:
            old_token = item.get('token', '')
            channel = item.get('channel', '')
            
            if channel in seen_sensors:
                # 如果该 channel 已存在，复用其 token
                new_token = seen_sensors[channel]
            else:
                # 如果是新 channel，创建新条目
                new_token = self.generate_new_token(old_token, 'sensor')
                item['token'] = new_token
                merged_data['sensor'].append(item)
                seen_sensors[channel] = new_token
            # 记录映射：旧token -> 新token (或复用的token)
            # 这样后续合并 calibrated_sensor 时，就能通过 old_token 找到这个公用的 new_token
            self.token_mapping['sensor'][old_token] = new_token
        print(f"  sensor: {len(merged_data['sensor'])} 项")
    
    def _merge_dependent_data(self, all_raw_data, merged_data):
        """合并有依赖关系的数据"""
        print("合并有依赖关系的数据...")
        
        # 合并instance（依赖category）
        print("合并instance数据...")
        instance_token_map = {}  # 记录旧token到新token的映射
        category_tokens = {cat['token'] for cat in merged_data['category']}
        
        invalid_category_count = 0
        for item in all_raw_data['instance']:
            old_token = item.get('token', '')
            new_token = self.generate_new_token(old_token, 'instance')
            item['token'] = new_token
            instance_token_map[old_token] = new_token
            
            # 【关键修复】更新category_token - 必须使用token_mapping
            old_category_token = item.get('category_token', '')
            if old_category_token:
                # 查找映射的新token
                if old_category_token in self.token_mapping['category']:
                    new_category_token = self.token_mapping['category'][old_category_token]
                    if new_category_token in category_tokens:
                        item['category_token'] = new_category_token
                    else:
                        invalid_category_count += 1
                        # 使用unknown类别作为备用
                        for cat in merged_data['category']:
                            if cat['name'] == 'unknown':
                                item['category_token'] = cat['token']
                                break
                else:
                    invalid_category_count += 1
                    # 如果映射不存在，也是错误
                    for cat in merged_data['category']:
                        if cat['name'] == 'unknown':
                            item['category_token'] = cat['token']
                            break
            
            merged_data['instance'].append(item)
        
        print(f"  instance: {len(merged_data['instance'])} 项 (发现 {invalid_category_count} 个无效category_token)")
        
        # 合并log, scene, map
        for data_type in ['log', 'scene', 'map']:
            for item in all_raw_data[data_type]:
                old_token = item.get('token', '')
                new_token = self.generate_new_token(old_token, data_type)
                item['token'] = new_token
                merged_data[data_type].append(item)
            print(f"  {data_type}: {len(merged_data[data_type])} 项")
        
        # 合并calibrated_sensor（依赖sensor）
        for item in all_raw_data['calibrated_sensor']:
            old_token = item.get('token', '')
            new_token = self.generate_new_token(old_token, 'calibrated_sensor')
            item['token'] = new_token
            
            # 更新sensor_token
            old_sensor_token = item.get('sensor_token', '')
            if old_sensor_token and old_sensor_token in self.token_mapping['sensor']:
                item['sensor_token'] = self.token_mapping['sensor'][old_sensor_token]
            
            merged_data['calibrated_sensor'].append(item)
        print(f"  calibrated_sensor: {len(merged_data['calibrated_sensor'])} 项")
        
        # 合并ego_pose
        for item in all_raw_data['ego_pose']:
            old_token = item.get('token', '')
            new_token = self.generate_new_token(old_token, 'ego_pose')
            item['token'] = new_token
            merged_data['ego_pose'].append(item)
        print(f"  ego_pose: {len(merged_data['ego_pose'])} 项")
        
        # 合并样本相关数据
        self._merge_sample_related_data(all_raw_data, merged_data, instance_token_map)
    
    def _merge_sample_related_data(self, all_raw_data, merged_data, instance_token_map):
        """合并样本相关数据（sample, sample_data, sample_annotation）"""
        # 【修复1】完整的三阶段方案：先创建映射，再更新引用，最后验证
        print("=== 开始合并样本相关数据 ===")
        
        # 1) 第一遍：预生成 token 映射（所有annotation）
        print("第一步：预生成 annotation token 映射...")
        annotation_token_map = {}
        for item in all_raw_data['sample_annotation']:
            old_token = item.get('token', '')
            if old_token:  # 【修复】只有非空token才映射
                new_token = str(uuid.uuid4())
                annotation_token_map[old_token] = new_token
                self.token_mapping['sample_annotation'][old_token] = new_token
        print(f"  生成了 {len(annotation_token_map)} 个 annotation token 映射")
        
        # 2) 预生成 sample 和 sample_data 映射
        print("第二步：预生成 sample 和 sample_data token 映射...")
        sample_token_map = {}
        for item in all_raw_data['sample']:
            old_token = item.get('token', '')
            if old_token:
                new_token = str(uuid.uuid4())
                sample_token_map[old_token] = new_token
                self.token_mapping['sample'][old_token] = new_token
        print(f"  生成了 {len(sample_token_map)} 个 sample token 映射")
        sample_data_token_map = {}
        for item in all_raw_data['sample_data']:
            old_token = item.get('token', '')
            if old_token:
                new_token = str(uuid.uuid4())
                sample_data_token_map[old_token] = new_token
                self.token_mapping['sample_data'][old_token] = new_token
        print(f"  生成了 {len(sample_data_token_map)} 个 sample_data token 映射")
        
        # 缓存其他映射
        ego_pose_mapping = self.token_mapping['ego_pose']
        calibrated_sensor_mapping = self.token_mapping['calibrated_sensor']
        scene_mapping = self.token_mapping['scene']
        visibility_mapping = self.token_mapping['visibility']
        attribute_mapping = self.token_mapping['attribute']
        global_instance_mapping = self.token_mapping.get('instance', {})

        # 3) 第二遍：构建 sample_annotation（使用正确的映射）
        print("第三步：构建 sample_annotation...")
        merged_data['sample_annotation'] = []
        ann_items_preserve_old_inst = []
        
        for item in all_raw_data['sample_annotation']:
            orig_token = item.get('token', '')
            
            # 【修复2】确保使用映射表中的token
            if orig_token in annotation_token_map:
                new_token = annotation_token_map[orig_token]
            else:
                # 备用方案：生成新token
                new_token = str(uuid.uuid4())
                if orig_token:
                    annotation_token_map[orig_token] = new_token
                    self.token_mapping['sample_annotation'][orig_token] = new_token

            old_inst = item.get('instance_token', '')
            new_inst_token = global_instance_mapping.get(old_inst, instance_token_map.get(old_inst, ''))

            new_ann = {
                'token': new_token,
                'sample_token': sample_token_map.get(item.get('sample_token', ''), ''),
                'instance_token': new_inst_token,
                'visibility_token': visibility_mapping.get(item.get('visibility_token', ''), item.get('visibility_token', '')),
                'translation': item.get('translation', [0,0,0]),
                'size': item.get('size', [0,0,0]),
                'rotation': item.get('rotation', [0,0,0,1]),
                'prev': annotation_token_map.get(item.get('prev', ''), ''),
                'next': annotation_token_map.get(item.get('next', ''), ''),
                'category_name': item.get('category_name', ''),
                'num_lidar_pts': item.get('num_lidar_pts', 0),
                'num_radar_pts': item.get('num_radar_pts', 0),
                'attribute_tokens': []
                # 'velocity': item.get('velocity', [0.0, 0.0])  # ✅ 保留 velocity
            }
            if 'attribute_tokens' in item and isinstance(item['attribute_tokens'], list):
                new_attrs = [attribute_mapping.get(a) for a in item['attribute_tokens'] if a and a in attribute_mapping]
                if new_attrs:
                    new_ann['attribute_tokens'] = new_attrs

            merged_data['sample_annotation'].append(new_ann)
            ann_items_preserve_old_inst.append((old_inst, orig_token, new_ann))

        print(f"  构建了 {len(merged_data['sample_annotation'])} 个 sample_annotation")

        # 4) 第三遍：构建 sample_data
        print("第四步：构建 sample_data...")
        merged_data['sample_data'] = []
        updated_sample_data_count = 0

        for item in all_raw_data['sample_data']:
            new_item = dict(item)
            old_tok = item.get('token', '')
            new_item['token'] = sample_data_token_map.get(old_tok, str(uuid.uuid4()))
            old_sample = item.get('sample_token', '')
            new_item['sample_token'] = sample_token_map.get(old_sample, '')
            if new_item['sample_token']:
                updated_sample_data_count += 1
            new_item['ego_pose_token'] = ego_pose_mapping.get(item.get('ego_pose_token', ''), item.get('ego_pose_token', ''))
            new_item['calibrated_sensor_token'] = calibrated_sensor_mapping.get(item.get('calibrated_sensor_token', ''), item.get('calibrated_sensor_token', ''))
            # 保留并映射 prev/next（支持 sweeps 链）
            new_item['prev'] = sample_data_token_map.get(item.get('prev', ''), '')
            new_item['next'] = sample_data_token_map.get(item.get('next', ''), '')
            new_item['is_key_frame'] = item.get('is_key_frame', True)
            merged_data['sample_data'].append(new_item)

        print(f"  构建了 {len(merged_data['sample_data'])} 个 sample_data")

        # 5) 第四遍：构建 sample，【关键】更新 anns 字段的映射
        print("第五步：构建 sample 并更新 anns 映射...")
        merged_data['sample'] = []
        anns_update_count = 0
        anns_empty_count = 0
        
        for item in all_raw_data['sample']:
            new_item = dict(item)
            old_tok = item.get('token', '')
            new_item['token'] = sample_token_map.get(old_tok, str(uuid.uuid4()))
            new_item['scene_token'] = scene_mapping.get(item.get('scene_token', ''), item.get('scene_token', ''))
            new_item['prev'] = ''
            new_item['next'] = ''
            
            if 'data' in item and isinstance(item['data'], dict):
                new_item['data'] = {ch: sample_data_token_map.get(tk, tk) for ch, tk in item['data'].items()}
            
            # 【关键修复】更新 anns 字段中的所有 token
            if 'anns' in item and isinstance(item['anns'], list):
                old_ann_tokens = item['anns']
                new_ann_tokens = []
                for old_ann_token in old_ann_tokens:
                    # 使用映射表查找新token
                    if old_ann_token in annotation_token_map:
                        new_ann_token = annotation_token_map[old_ann_token]
                        new_ann_tokens.append(new_ann_token)
                        anns_update_count += 1
                    else:
                        # 【调试】找不到映射的annotation
                        print(f"  警告: sample {old_tok[:8]}... 的anns中的token {old_ann_token[:8]}... 在映射表中未找到")
                        # 保留原始token作为备用（虽然可能无效）
                        # new_ann_tokens.append(old_ann_token)
                new_item['anns'] = new_ann_tokens
                if not new_ann_tokens:
                    anns_empty_count += 1
            else:
                new_item['anns'] = []
                
            merged_data['sample'].append(new_item)
        
        print(f"  构建了 {len(merged_data['sample'])} 个 sample")
        print(f"  更新了 {anns_update_count} 个 anns token")
        print(f"  发现 {anns_empty_count} 个没有有效anns的sample")

        print("=== 样本相关数据合并完成 ===\n")
        
        # 【新增】验证 anns 的一致性
        self._verify_anns_consistency(merged_data, annotation_token_map)
        
        # 5) 恢复原始链接（如之前所示）
        print("  使用原始 annotation prev/next 链恢复（若存在）...")
        orig_to_new = {orig: new for (_, orig, new) in ann_items_preserve_old_inst if orig}
        orig_next = {item.get('token',''): item.get('next','') for item in all_raw_data['sample_annotation'] if item.get('token','')}
        visited = set()
        restored = 0
        for orig_start in list(orig_to_new.keys()):
            if orig_start in visited:
                continue
            cur = orig_start
            chain = []
            while cur and cur not in visited:
                visited.add(cur)
                new_ann = orig_to_new.get(cur)
                if new_ann:
                    chain.append(new_ann)
                cur = orig_next.get(cur, '')
            if chain:
                for i, ann in enumerate(chain):
                    prev_t = chain[i-1]['token'] if i>0 else ''
                    next_t = chain[i+1]['token'] if i < len(chain)-1 else ''
                    if ann.get('prev','') != prev_t:
                        ann['prev'] = prev_t; restored += 1
                    if ann.get('next','') != next_t:
                        ann['next'] = next_t; restored += 1
        print(f"  原始链恢复，填充/修正 {restored} 个 prev/next 字段")

        # 6) fallback 重建
        #注释 因这会破坏上游生成的“速度熔断”逻辑，导致异常速度
        # print("  对剩余项按原始 old_instance 分组并按 sample.timestamp 重建...")
        # sample_ts = {s['token']: s.get('timestamp', 0) for s in merged_data['sample']}
        # groups = defaultdict(list)
        # for old_inst, orig_tok, new_ann in ann_items_preserve_old_inst:
        #     if new_ann.get('prev') or new_ann.get('next'):
        #         continue
        #     ts = sample_ts.get(new_ann.get('sample_token',''), 0)
        #     groups[old_inst].append((ts, new_ann))
        # filled = 0
        # for old_inst, lst in groups.items():
        #     if len(lst) <= 1:
        #         if lst:
        #             lst[0][1]['prev'] = ''; lst[0][1]['next'] = ''
        #         continue
        #     mapped_new_inst = global_instance_mapping.get(old_inst, instance_token_map.get(old_inst, ''))
        #     if mapped_new_inst:
        #         for _, ann in lst:
        #             ann['instance_token'] = mapped_new_inst
        #     lst.sort(key=lambda x: x[0])
        #     for i, (_, ann) in enumerate(lst):
        #         prev_t = lst[i-1][1]['token'] if i>0 else ''
        #         next_t = lst[i+1][1]['token'] if i < len(lst)-1 else ''
        #         if ann.get('prev','') != prev_t:
        #             ann['prev'] = prev_t; filled += 1
        #         if ann.get('next','') != next_t:
        #             ann['next'] = next_t; filled += 1
        # print(f"  fallback 重建完成，补充/修正 {filled} 个 prev/next 字段")

    def _verify_anns_consistency(self, merged_data, annotation_token_map):
        """【新增】验证 sample.anns 中的所有token都在 sample_annotation 中"""
        print("验证 sample.anns 一致性...")
        
        # 构建 annotation token 集合
        ann_tokens_set = {ann['token'] for ann in merged_data['sample_annotation']}
        
        invalid_anns_count = 0
        missing_anns_count = 0
        samples_with_invalid_anns = []
        
        for sample in merged_data['sample']:
            anns = sample.get('anns', [])
            for ann_token in anns:
                if ann_token not in ann_tokens_set:
                    invalid_anns_count += 1
                    if len(samples_with_invalid_anns) < 5:  # 只记录前5个
                        samples_with_invalid_anns.append({
                            'sample_token': sample['token'][:8],
                            'invalid_ann_token': ann_token[:8]
                        })
        
        # 检查是否有sample没有anns
        for sample in merged_data['sample']:
            if not sample.get('anns', []):
                missing_anns_count += 1
        
        print(f"  总样本数: {len(merged_data['sample'])}")
        print(f"  总annotation数: {len(ann_tokens_set)}")
        print(f"  无效的anns token: {invalid_anns_count}")
        print(f"  没有anns的sample: {missing_anns_count}")
        print(f"  anns-annotation映射: {sum(1 for s in merged_data['sample'] if s.get('anns', []))} 个sample有anns")
        
        if invalid_anns_count > 0:
            print(f"\n  【错误】发现 {invalid_anns_count} 个无效的anns token:")
            for item in samples_with_invalid_anns:
                print(f"    sample {item['sample_token']}... 引用了不存在的 {item['invalid_ann_token']}...")
            return False
        else:
            print("  ✓ 所有 anns token 都有效")
            return True
    
    def _validate_sample_data_refs(self, merged_data, sample_data_token_map):
        """验证sample和sample_data之间的引用关系"""
        print("验证sample和sample_data之间的引用关系...")
        
        # 构建查找表
        sample_data_tokens = {sd['token'] for sd in merged_data['sample_data']}
        sample_tokens = {s['token'] for s in merged_data['sample']}
        
        # 验证1：sample中的data字段引用是否有效
        invalid_refs = 0
        for sample in merged_data['sample']:
            if 'data' in sample and isinstance(sample['data'], dict):
                for channel, data_token in sample['data'].items():
                    if data_token not in sample_data_tokens:
                        invalid_refs += 1
                        print(f"错误: sample {sample['token']} 的通道 {channel} 引用了无效的sample_data token: {data_token}")
        
        if invalid_refs == 0:
            print("  所有sample的data字段引用都有效")
        else:
            print(f"  发现 {invalid_refs} 个无效的sample_data引用")
        
        # 验证2：sample_data中的sample_token是否有效
        invalid_sample_refs = 0
        for sd in merged_data['sample_data']:
            sample_token = sd.get('sample_token', '')
            if sample_token and sample_token not in sample_tokens:
                invalid_sample_refs += 1
                print(f"错误: sample_data {sd['token']} 引用了无效的sample_token: {sample_token}")
        
        if invalid_sample_refs == 0:
            print("  所有sample_data的sample_token引用都有效")
        else:
            print(f"  发现 {invalid_sample_refs} 个无效的sample_token引用")
        
        # 验证3：检查LIDAR_TOP通道
        missing_lidar = []
        for sample in merged_data['sample']:
            if 'data' in sample and isinstance(sample['data'], dict):
                if 'LIDAR_TOP' not in sample['data']:
                    missing_lidar.append(sample['token'])
        
        if missing_lidar:
            print(f"警告: 发现 {len(missing_lidar)} 个sample缺少LIDAR_TOP数据")
    def _fix_all_references(self, merged_data):
        """修复所有跨数据类型的引用关系（高效版）"""
        print("修复引用关系（优化版）...")
        # 预构建查找表
        sample_list = merged_data['sample']
        sample_data_list = merged_data['sample_data']
        ann_list = merged_data['sample_annotation']
        inst_list = merged_data['instance']
        scene_list = merged_data['scene']
        
        sample_token_to_sample = {s['token']: s for s in sample_list}
        sample_data_token_to_obj = {sd['token']: sd for sd in sample_data_list}
        ann_token_to_obj = {a['token']: a for a in ann_list}
        inst_token_to_obj = {i['token']: i for i in inst_list}
        
        # 1) 修复 sample 的 prev/next：按 scene 分组，一次排序并就地设置
        print(" 修复 sample.prev/next ...")
        samples_by_scene = defaultdict(list)
        for s in sample_list:
            samples_by_scene[s.get('scene_token', '')].append(s)
        fixed = 0
        for scene_token, sl in samples_by_scene.items():
            if len(sl) <= 1:
                if sl:
                    sl[0]['prev'] = ''
                    sl[0]['next'] = ''
                continue
            sl.sort(key=lambda x: x.get('timestamp', 0))
            for i, s in enumerate(sl):
                prev_t = sl[i-1]['token'] if i > 0 else ''
                next_t = sl[i+1]['token'] if i < len(sl)-1 else ''
                if s.get('prev') != prev_t:
                    s['prev'] = prev_t
                if s.get('next') != next_t:
                    s['next'] = next_t
                fixed += 1
        print(f"  完成 sample 链接设置，处理 {len(sample_list)} 个 sample（按 scene 分组）")
        
        # 2) 修复 sample_data 的 prev/next：按 (sample_token, channel) 分组
        print(" 修复 sample_data.prev/next ...")
        sd_groups = defaultdict(list)
        for sd in sample_data_list:
            key = sd.get('calibrated_sensor_token', '')
            sd_groups[key].append(sd)
        for key, group in sd_groups.items():
            if len(group) <= 1:
                if group:
                    group[0]['prev'] = ''
                    group[0]['next'] = ''
                continue
            group.sort(key=lambda x: x.get('timestamp', 0))
            for i, sd in enumerate(group):
                sd['prev'] = group[i-1]['token'] if i > 0 else ''
                sd['next'] = group[i+1]['token'] if i < len(group)-1 else ''
        print(f"  完成 sample_data 链接设置，组数 {len(sd_groups)}")
        
        # 3) 验证并尽量保留已有 annotation prev/next（若映射后存在）
        print(" 验证 annotation 现有 prev/next ...")
        ann_tokens = set(ann_token_to_obj.keys())
        # 清除指向不存在 token 的 prev/next（快速检查）
        invalid_clears = 0
        for a in ann_list:
            p = a.get('prev', '')
            n = a.get('next', '')
            if p and p not in ann_tokens:
                a['prev'] = ''
                invalid_clears += 1
            if n and n not in ann_tokens:
                a['next'] = ''
                invalid_clears += 1
        print(f"  清除了 {invalid_clears} 个指向不存在 annotation 的 prev/next")
        
        # 4) 在 instance 级别补全和修正 annotation 链：按 instance 分组并按 sample.timestamp 排序
        print(" 在 instance 级别补全 annotation 链 ...")
        
        # 关键修复：按 instance 而非全局时间序列
        # anns_by_inst = defaultdict(list)
        # for a in ann_list:
        #     inst_tok = a.get('instance_token', '')
        #     if inst_tok:  # 只处理有 instance_token 的
        #         anns_by_inst[inst_tok].append(a)
        
        # filled = 0
        # for inst_tok, al in anns_by_inst.items():
        #     if len(al) <= 1:
        #         if al:
        #             al[0]['prev'] = ''
        #             al[0]['next'] = ''
        #         continue
            
        #     # 获取每个 annotation 的 sample timestamp
        #     annotated = []
        #     for a in al:
        #         st = 0
        #         stoken = a.get('sample_token', '')
        #         s = sample_token_to_sample.get(stoken)
        #         if s:
        #             st = s.get('timestamp', 0)
        #         annotated.append((st, a))
            
        #     annotated.sort(key=lambda x: x[0])
            
        #     # 为同一实例的相邻 annotation 设置 prev/next
        #     for i, (_, a) in enumerate(annotated):
        #         prev_t = annotated[i-1][1]['token'] if i > 0 else ''
        #         next_t = annotated[i+1][1]['token'] if i < len(annotated)-1 else ''
                
        #         if a.get('prev') != prev_t:
        #             a['prev'] = prev_t
        #             filled += 1
        #         if a.get('next') != next_t:
        #             a['next'] = next_t
        #             filled += 1
        
        # print(f"  补全/修正了约 {filled} 个 annotation 的 prev/next")
        
        # 5) 修复 scene.first/last 与 instance first/last、nbr_annotations
        print(" 修复 scene、instance 的 first/last/nbr ...")
        # 重新构建 anns_by_inst (只读，用于统计 instance 信息)
        anns_by_inst = defaultdict(list)
        for a in ann_list:
            inst_tok = a.get('instance_token', '')
            if inst_tok:
                anns_by_inst[inst_tok].append(a)
        # scene -> samples 已有 samples_by_scene
        for scene in scene_list:
            st = scene.get('token', '')
            s_list = samples_by_scene.get(st, [])
            if s_list:
                s_list.sort(key=lambda x: x.get('timestamp', 0))
                scene['first_sample_token'] = s_list[0]['token']
                scene['last_sample_token'] = s_list[-1]['token']
            else:
                scene['first_sample_token'] = ''
                scene['last_sample_token'] = ''
        # instance -> annotations
        for inst in inst_list:
            it = inst.get('token', '')
            al = anns_by_inst.get(it, [])
            if al:
                # 使用 sample timestamp 排序
                tmp = []
                for a in al:
                    s = sample_token_to_sample.get(a.get('sample_token', ''))
                    ts = s.get('timestamp', 0) if s else 0
                    tmp.append((ts, a))
                tmp.sort(key=lambda x: x[0])
                inst['first_annotation_token'] = tmp[0][1]['token']
                inst['last_annotation_token'] = tmp[-1][1]['token']
                inst['nbr_annotations'] = len(al)
            else:
                inst['first_annotation_token'] = ''
                inst['last_annotation_token'] = ''
                inst['nbr_annotations'] = 0
        print("  引用修复完成")
    
    def _validate_data_integrity(self, merged_data):
        """验证数据完整性"""
        print("验证数据完整性...")
        
        # 查找或创建unknown类别
        unknown_category = None
        for cat in merged_data['category']:
            if cat['name'] == 'unknown':
                unknown_category = cat
                break
        
        if not unknown_category:
            unknown_category = {
                "token": str(uuid.uuid4()),
                "name": "unknown",
                "description": "Unknown category for instances with missing category information"
            }
            merged_data['category'].append(unknown_category)
        
        # 【优化】使用集合缓存，避免重复遍历
        category_tokens = {cat['token'] for cat in merged_data['category']}
        instance_tokens = {inst['token'] for inst in merged_data['instance']}
        sample_tokens = {s['token'] for s in merged_data['sample']}
        attribute_tokens = {attr['token'] for attr in merged_data['attribute']}
        
        print(f"  检查数据状态: category={len(category_tokens)}, instance={len(instance_tokens)}, sample={len(sample_tokens)}")
        
        # 【新增】检查sample_annotation中的关键字段
        print("检查sample_annotation字段完整性...")
        missing_fields = {'category_name': 0, 'num_lidar_pts': 0, 'num_radar_pts': 0}
        
        for annotation in merged_data['sample_annotation']:
            for field in missing_fields.keys():
                if field not in annotation:
                    missing_fields[field] += 1
        
        for field, count in missing_fields.items():
            if count > 0:
                print(f"警告: 发现 {count} 个sample_annotation缺少 {field} 字段")
            else:
                print(f"  所有sample_annotation都包含 {field} 字段")
        
        # 检查instance中的category_token
        instance_issues = []
        for instance in merged_data['instance']:
            category_token = instance.get('category_token', '')
            if not category_token or category_token not in category_tokens:
                instance_issues.append((instance['token'], category_token))
                instance['category_token'] = unknown_category['token']
        
        if instance_issues:
            print(f"警告: 发现 {len(instance_issues)} 个instance的category_token无效")
            # 只显示前5个
            for i, (inst_token, cat_token) in enumerate(instance_issues[:5]):
                print(f"  例子 {i+1}: instance={inst_token[:8]}... category_token={cat_token[:8] if cat_token else 'EMPTY'}...")
        
        # 检查sample_annotation中的instance_token和sample_token
        annotation_instance_issues = []
        annotation_sample_issues = []
        
        for annotation in merged_data['sample_annotation']:
            instance_token = annotation.get('instance_token', '')
            if not instance_token or instance_token not in instance_tokens:
                annotation_instance_issues.append(annotation['token'])
                if merged_data['instance']:
                    annotation['instance_token'] = merged_data['instance'][0]['token']
            
            sample_token = annotation.get('sample_token', '')
            if not sample_token or sample_token not in sample_tokens:
                annotation_sample_issues.append(annotation['token'])
        
        if annotation_instance_issues:
            print(f"警告: 发现 {len(annotation_instance_issues)} 个sample_annotation的instance_token无效，已修复")
        if annotation_sample_issues:
            print(f"警告: 发现 {len(annotation_sample_issues)} 个sample_annotation的sample_token无效")
        
        # 检查scene中的first_sample_token和last_sample_token
        scene_issues = []
        for scene in merged_data['scene']:
            for field in ['first_sample_token', 'last_sample_token']:
                token = scene.get(field, '')
                if token and token not in sample_tokens:
                    scene_issues.append((scene['token'], field, token))
        
        if scene_issues:
            print(f"警告: 发现 {len(scene_issues)} 个scene的first/last_sample_token无效")
        
        # 检查attribute_tokens的有效性
        attribute_issues = []
        for annotation in merged_data['sample_annotation']:
            attr_tokens = annotation.get('attribute_tokens', [])
            for attr_token in attr_tokens:
                if attr_token and attr_token not in attribute_tokens:
                    attribute_issues.append(annotation['token'])
                    break
        
        if attribute_issues:
            print(f"警告: 发现 {len(attribute_issues)} 个sample_annotation的attribute_token无效")
        
        print("数据完整性验证完成")
        # 【新增】检查annotation的prev/next链接
        print("检查annotation的prev/next链接...")
        valid_prev_next_pairs = 0
        invalid_prev_next_pairs = 0
        
        # 创建annotation token到annotation的映射
        ann_by_token = {ann['token']: ann for ann in merged_data['sample_annotation']}
        
        for ann in merged_data['sample_annotation']:
            prev_token = ann.get('prev', '')
            next_token = ann.get('next', '')
            
            # 检查prev是否指向有效annotation
            if prev_token:
                prev_ann = ann_by_token.get(prev_token)
                if prev_ann and prev_ann.get('next') == ann['token']:
                    valid_prev_next_pairs += 1
                else:
                    invalid_prev_next_pairs += 1
                    # 只在调试时打印前几个错误
                    if invalid_prev_next_pairs <= 3:
                        print(f"  警告: annotation {ann['token'][:8]}... 的prev链接不一致")
                        if prev_ann:
                            print(f"    prev annotation的next是: {prev_ann.get('next', '空')[:8] if prev_ann.get('next') else '空'}...")
            
            # 检查next是否指向有效annotation
            if next_token:
                next_ann = ann_by_token.get(next_token)
                if next_ann and next_ann.get('prev') == ann['token']:
                    valid_prev_next_pairs += 1
                else:
                    invalid_prev_next_pairs += 1
                    # 只在调试时打印前几个错误
                    if invalid_prev_next_pairs <= 3:
                        print(f"  警告: annotation {ann['token'][:8]}... 的next链接不一致")
                        if next_ann:
                            print(f"    next annotation的prev是: {next_ann.get('prev', '空')[:8] if next_ann.get('prev') else '空'}...")
        
        total_annotations = len(merged_data['sample_annotation'])
        annotations_with_prev = sum(1 for ann in merged_data['sample_annotation'] if ann.get('prev', ''))
        annotations_with_next = sum(1 for ann in merged_data['sample_annotation'] if ann.get('next', ''))
        
        print(f"  annotation总数: {total_annotations}")
        print(f"  有prev的annotation: {annotations_with_prev}")
        print(f"  有next的annotation: {annotations_with_next}")
        print(f"  有效prev/next链接对: {valid_prev_next_pairs}")
        print(f"  无效prev/next链接: {invalid_prev_next_pairs}")

    def _save_merged_data(self, merged_data, output_folder):
        """保存合并后的数据"""
        # 保存token映射用于调试（只写一次）
        debug_info = {
            'mapping_summary': {k: len(v) for k, v in self.token_mapping.items()},
            'sample': self.token_mapping.get('sample', {}),
            'sample_data': self.token_mapping.get('sample_data', {}),
            'sample_annotation': self.token_mapping.get('sample_annotation', {})
        }
        with open(os.path.join(output_folder, '_debug_token_map.json'), 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, indent=2, sort_keys=True, ensure_ascii=False)
        
        # 保存各数据文件
        for data_type, data_list in merged_data.items():
            output_file = os.path.join(output_folder, f"{data_type}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=2, ensure_ascii=False)
            print(f"已保存 {data_type}.json, 包含 {len(data_list)} 个条目")

def main():
    parser = argparse.ArgumentParser(description='合并多个nuScenes格式数据集')
    parser.add_argument('--datasets', nargs='+', required=True, 
                       help='要合并的数据集文件夹路径列表')
    parser.add_argument('--output', required=True, 
                       help='合并后的输出文件夹路径')
    
    args = parser.parse_args()
    
    # 验证输入
    for dataset in args.datasets:
        if not os.path.exists(dataset):
            print(f"错误: 数据集文件夹不存在: {dataset}")
            return
    
    # 执行合并
    merger = NuScenesMerger()
    merger.merge_datasets(args.datasets, args.output)

if __name__ == "__main__":
    # 使用示例python merge_nuscenes.py --datasets /path/to/dataset1 /path/to/dataset2 
    # --output /path/to/merged_output
    # （取消注释下面的代码并注释main()调用来直接运行）
    # datasets = [
    #     "/path/to/dataset1",
    #     "/path/to/dataset2", 
    #     "/path/to/dataset3"
    # ]
    # output = "/path/to/merged_dataset"
    
    # merger = NuScenesMerger()
    # merger.merge_datasets(datasets, output)
    
    main()
