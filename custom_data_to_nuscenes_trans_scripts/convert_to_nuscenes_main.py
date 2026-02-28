
import json
import os
import uuid
import numpy as np
from datetime import datetime
import argparse

def load_custom_annotations(file_path):
    """加载自定义标注文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_nuscenes_categories():
    """创建NuScenes类别"""
    categories = [
        {"token": str(uuid.uuid4()), "name": "human.pedestrian.adult", "description": "Adult pedestrian"},
        {"token": str(uuid.uuid4()), "name": "human.pedestrian.child", "description": "Child pedestrian"},
        {"token": str(uuid.uuid4()), "name": "human.pedestrian.wheelchair", "description": "Pedestrian in wheelchair"},
        {"token": str(uuid.uuid4()), "name": "human.pedestrian.stroller", "description": "Pedestrian with stroller"},
        {"token": str(uuid.uuid4()), "name": "human.pedestrian.personal_mobility", "description": "Personal mobility device"},
        {"token": str(uuid.uuid4()), "name": "human.pedestrian.police_officer", "description": "Police officer"},
        {"token": str(uuid.uuid4()), "name": "human.pedestrian.construction_worker", "description": "Construction worker"},
        {"token": str(uuid.uuid4()), "name": "animal", "description": "Animal"},
        {"token": str(uuid.uuid4()), "name": "vehicle.car", "description": "Car"},
        {"token": str(uuid.uuid4()), "name": "vehicle.motorcycle", "description": "Motorcycle"},
        {"token": str(uuid.uuid4()), "name": "vehicle.bicycle", "description": "Bicycle"},
        {"token": str(uuid.uuid4()), "name": "vehicle.bus.bendy", "description": "Bendy bus"},
        {"token": str(uuid.uuid4()), "name": "vehicle.bus.rigid", "description": "Rigid bus"},
        {"token": str(uuid.uuid4()), "name": "vehicle.truck", "description": "Truck"},
        {"token": str(uuid.uuid4()), "name": "vehicle.construction", "description": "Construction vehicle"},
        {"token": str(uuid.uuid4()), "name": "vehicle.emergency.ambulance", "description": "Ambulance"},
        {"token": str(uuid.uuid4()), "name": "vehicle.emergency.police", "description": "Police vehicle"},
        {"token": str(uuid.uuid4()), "name": "vehicle.emergency.fire", "description": "Fire truck"},
        {"token": str(uuid.uuid4()), "name": "vehicle.trailer", "description": "Trailer"},
        {"token": str(uuid.uuid4()), "name": "movable_object.barrier", "description": "Barrier"},
        {"token": str(uuid.uuid4()), "name": "movable_object.trafficcone", "description": "Traffic cone"},
        {"token": str(uuid.uuid4()), "name": "movable_object.pushable_pullable", "description": "Pushable or pullable object"},
        {"token": str(uuid.uuid4()), "name": "movable_object.debris", "description": "Debris"},
        {"token": str(uuid.uuid4()), "name": "static_object.bicycle_rack", "description": "Bicycle rack"}
    ]

    return categories

def extract_english_category(label_name):
    """从中文/混合标签中提取规范化英文类别（返回 canonical key，如 'bus','car'）"""
    import re
    if not label_name:
        return ""
    s = str(label_name).lower().strip()
    # 去掉“复制xx”后缀/括号
    s = re.sub(r'[（(]?复制\d+[)）]?', '', s)

    m = re.search(r'[（(]([^）)]+)[）)]', s)
    if m:
        inner = m.group(1).strip().lower()
        inner_clean = re.sub(r'[^a-z0-9]', '', inner)
        if inner_clean in ("constructionvehicle", "constructionveh", "constructioncar"):
            return "construction"
        if inner_clean:
            return inner_clean

    # 同义词表（优先级定义：bus 前于 car）
    synonyms = {
        "bus": ["bus", "巴士", "公交", "公交车", "大巴", "客车", "客运车"],
        "truck": ["truck", "卡车", "货车"],
        "motorcycle": ["motorcycle", "摩托车", "摩托","两轮车辆", 
                    "cyclelist", "两轮车辆", "电动车", "电动自行车", "三轮车"],
        "bicycle": ["bicycle", "自行车", "单车", "两轮车辆（bicycle）"],
        "car": ["car", "汽车", "轿车", "乘用车", "小型乘用车（car）",
                "商用车辆", "commercial vehicle"],
        "pedestrian": ["pedestrian", "行人", "人"],
        "animal": ["animal", "动物",],
        "barrier": ["barrier", "护栏", "栏杆", "concrete_barrier", 
                    "concretebarrier", ],
        "cone": ["cone", "锥桶", "路锥", "锥",],
        "trailer": ["trailer", "拖车"],
        "construction": ["construction", "施工", "工程","工程车",
                         "施工车", "special vehicle", "特种车辆"
                         "construction vehicle", "constructionvehicle"],
        "debris": ["debris", "散落", "路面散落","路面散落障碍物（debris）"]
    }

    priority = ["bus", "truck", "motorcycle", "bicycle", "car",
                "pedestrian", "animal", "barrier", "cone", "trailer", "construction", "debris"]

    for key in priority:
        for kw in synonyms.get(key, []):
            if kw in s:
                return key

    # 尝试提取ASCII单词再匹配
    words = re.findall(r'[a-z0-9]+', s)
    for w in words:
        for key, keys in synonyms.items():
            if w in keys:
                return key

    return s  # 回退原始小写字符串

def create_custom_to_nuscenes_mapping(custom_labels):
    """创建以 str(label_id) 为键的映射，值为 NuScenes 类别名；打印每条映射便于调试"""
    custom_to_nuscenes = {}
    print("\n=== 创建标签映射开始 ===")
    if not custom_labels:
        print("警告: custom_labels 为空")
        return custom_to_nuscenes

    # 用于去重：跟踪已处理的label text，避免重复映射
    processed_label_texts = set()
    stats = {}
    
    for i, lbl in enumerate(custom_labels):
        # wrong: 不再使用 lid = lbl.get("id")
        # right: 改用 label text 作为键
        label_text = str(lbl.get("label", "")).strip().lower()
        
        # 去重：同一个label text只处理一次
        if label_text in processed_label_texts:
            continue
        processed_label_texts.add(label_text)
        
        raw = label_text
        eng = extract_english_category(raw)

        # 精确映射（使用eng）
        if eng == "bus":
            nus_cat = "vehicle.bus.rigid"
        elif eng == "truck":
            nus_cat = "vehicle.truck"
        elif eng == "motorcycle" or eng == "cyclelist":
            nus_cat = "vehicle.motorcycle"
        elif eng == "bicycle":
            nus_cat = "vehicle.bicycle"
        elif eng == "pedestrian":
            nus_cat = "human.pedestrian.adult"
        elif eng == "animal":
            nus_cat = "animal"
        elif eng == "barrier" or eng == "concrete_barrier" \
                or eng == "concretebarrier":
            nus_cat = "movable_object.barrier"
        elif eng == "cone":
            nus_cat = "movable_object.trafficcone"
        elif eng == "trailer":
            nus_cat = "vehicle.trailer"
        elif eng == "construction" or eng.startswith("construction") \
                or eng == "specialvehicle":
            nus_cat = "vehicle.construction"
        elif eng == "car" or eng == "commercialvehicle":
            nus_cat = "vehicle.car"
        elif "debris" in eng:
            nus_cat = "movable_object.debris"
        else:
            nus_cat = None   # 未识别直接跳过
        if nus_cat is None:
            print(f"跳过未识别标签: label_text='{label_text}', 提取='{eng}'")
            continue

         # 改用 label_text 作为键
        custom_to_nuscenes[label_text] = nus_cat
        stats.setdefault(nus_cat, 0)
        stats[nus_cat] += 1

        print(f"  label_text='{label_text}' -> extracted='{eng}' -> mapped='{nus_cat}'")

    print("=== 映射统计 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("=== 创建标签映射完成 ===\n")
    return custom_to_nuscenes

def create_nuscenes_attributes():
    """创建NuScenes属性"""
    attributes = [
        {"token": str(uuid.uuid4()), "name": "cycle.with_rider", "description": "Cycle has a rider"},
        {"token": str(uuid.uuid4()), "name": "cycle.without_rider", "description": "Cycle has no rider"},
        {"token": str(uuid.uuid4()), "name": "pedestrian.moving", "description": "Pedestrian is moving"},
        {"token": str(uuid.uuid4()), "name": "pedestrian.standing", "description": "Pedestrian is standing"},
        {"token": str(uuid.uuid4()), "name": "pedestrian.sitting_lying_down", "description": "Pedestrian is sitting or lying down"},
        {"token": str(uuid.uuid4()), "name": "vehicle.moving", "description": "Vehicle is moving"},
        {"token": str(uuid.uuid4()), "name": "vehicle.stopped", "description": "Vehicle is stopped"},
        {"token": str(uuid.uuid4()), "name": "vehicle.parked", "description": "Vehicle is parked"},
        {"token": str(uuid.uuid4()), "name": "bus.rigid", "description": "Bus is rigid"},
        {"token": str(uuid.uuid4()), "name": "bus.bendy", "description": "Bus is bendy"},
        {"token": str(uuid.uuid4()), "name": "truck.trailer", "description": "Truck with trailer"},
        {"token": str(uuid.uuid4()), "name": "construction", "description": "Construction vehicle"},
        {"token": str(uuid.uuid4()), "name": "emergency", "description": "Emergency vehicle"},
        {"token": str(uuid.uuid4()), "name": "ignore", "description": "Ignore this annotation"}
    ]

    return attributes

def create_nuscenes_visibility():
    """创建NuScenes可见性"""
    visibility = [
        {"token": "1", "description": "visibility: 0-40%", "level": "v0-40"},
        {"token": "2", "description": "visibility: 40-60%", "level": "v40-60"},
        {"token": "3", "description": "visibility: 60-80%", "level": "v60-80"},
        {"token": "4", "description": "visibility: 80-100%", "level": "v80-100"}
    ]

    return visibility

def create_nuscenes_sensors():
    """创建NuScenes传感器"""
    sensors = [
        {"token": str(uuid.uuid4()), "channel": "CAM_FRONT", "modality": "camera"},
        {"token": str(uuid.uuid4()), "channel": "CAM_FRONT_LEFT", "modality": "camera"},
        {"token": str(uuid.uuid4()), "channel": "CAM_FRONT_RIGHT", "modality": "camera"},
        {"token": str(uuid.uuid4()), "channel": "CAM_BACK", "modality": "camera"},
        {"token": str(uuid.uuid4()), "channel": "CAM_BACK_LEFT", "modality": "camera"},
        {"token": str(uuid.uuid4()), "channel": "CAM_BACK_RIGHT", "modality": "camera"},
        {"token": str(uuid.uuid4()), "channel": "LIDAR_TOP", "modality": "lidar"}
    ]

    return sensors

def load_camera_calibration(camera_calib_path):
    """加载相机标定文件"""
    camera_calib = {}

    # 加载前相机标定
    with open(os.path.join(camera_calib_path, 'camera1_front.json'), 'r') as f:
        front_calib = json.load(f)
        camera_calib['front'] = front_calib

    # 加载左前相机标定 - 在NuScenes中是后左相机
    with open(os.path.join(camera_calib_path, 'camera3_front_left.json'), 'r') as f:
        left_front_calib = json.load(f)
        camera_calib['front_left'] = left_front_calib  # 实际对应NuScenes的BACK_LEFT

    # 加载右前相机标定 - 在NuScenes中是后右相机
    with open(os.path.join(camera_calib_path, 'camera4_front_right.json'), 'r') as f:
        right_front_calib = json.load(f)
        camera_calib['front_right'] = right_front_calib  # 实际对应NuScenes的BACK_RIGHT

    # 加载后相机标定
    with open(os.path.join(camera_calib_path, 'camera2_rear.json'), 'r') as f:
        back_calib = json.load(f)
        camera_calib['rear'] = back_calib

    # 加载左后相机标定 - 在NuScenes中是前左相机
    with open(os.path.join(camera_calib_path, 'camera5_rear_left.json'), 'r') as f:
        left_back_calib = json.load(f)
        camera_calib['rear_left'] = left_back_calib  # 实际对应NuScenes的FRONT_LEFT

    # 加载右后相机标定 - 在NuScenes中是前右相机
    with open(os.path.join(camera_calib_path, 'camera6_rear_right.json'), 'r') as f:
        right_back_calib = json.load(f)
        camera_calib['rear_right'] = right_back_calib  # 实际对应NuScenes的FRONT_RIGHT

    # 打印调试信息
    print("标定文件加载结果:")
    for key, value in camera_calib.items():
        print(f"  {key}: 已加载")

    return camera_calib

def extrinsic_matrix_to_translation_rotation(extrinsic_matrix):
    """将外参矩阵转换为平移和旋转四元数 - 彻底修正版本"""
    # print(f"原始外参矩阵长度: {len(extrinsic_matrix)}")
    # print(f"原始外参矩阵: {extrinsic_matrix}")
    # 处理16元素的外参矩阵 - 正确的处理方式
    if len(extrinsic_matrix) == 16:
        print("检测到16元素外参矩阵，重新构造3x4矩阵")
        # 16元素的外参矩阵实际上是4x4齐次变换矩阵
        # 我们需要将其转换为3x4的[R|t]形式
        matrix_4x4 = np.array(extrinsic_matrix).reshape(4, 4)
        # 提取旋转和平移部分，忽略最后一行[0,0,0,1]
        rotation_matrix = matrix_4x4[:3, :3]
        translation = matrix_4x4[:3, 3].tolist()
        
        # print(f"从4x4矩阵提取的旋转矩阵:\n{rotation_matrix}")
        # print(f"从4x4矩阵提取的平移: {translation}")
    elif len(extrinsic_matrix) == 12:
        # 12元素的外参矩阵已经是3x4的[R|t]形式
        matrix_3x4 = np.array(extrinsic_matrix).reshape(3, 4)
        rotation_matrix = matrix_3x4[:, :3]
        translation = matrix_3x4[:, 3].tolist()
        
        # print(f"从3x4矩阵提取的旋转矩阵:\n{rotation_matrix}")
        # print(f"从3x4矩阵提取的平移: {translation}")
    else:
        raise ValueError(f"外参矩阵应该有12或16个元素，但得到 {len(extrinsic_matrix)}")
    # 方式1：直接使用（不取逆）
    use_direct = False  # 测试
    
    if use_direct:
        final_rotation = rotation_matrix
        final_translation = translation
        print("使用直接变换（不取逆）")
    else:
        # 方式2：取逆（原来的方式）
        transform_4x4 = np.eye(4)
        transform_4x4[:3, :3] = rotation_matrix
        transform_4x4[:3, 3] = translation
        inv_transform = np.linalg.inv(transform_4x4)
        final_rotation = inv_transform[:3, :3]
        final_translation = inv_transform[:3, 3].tolist()
        print("使用逆变换")
    
    # 将旋转矩阵转换为四元数
    from scipy.spatial.transform import Rotation as R
    rotation_obj = R.from_matrix(final_rotation)
    quaternion = rotation_obj.as_quat()  # [x, y, z, w]
    
    rotation = [float(quaternion[3]), float(quaternion[0]), float(quaternion[1]),
               float(quaternion[2]) ]
    
    # print(f"最终平移: {final_translation}")
    # print(f"最终四元数: {rotation}")
    
    return final_translation, rotation
   
def create_nuscenes_calibrated_sensors(sensors, camera_calib_path):
    """创建NuScenes校准传感器"""
    calibrated_sensors = []

    # 加载相机标定数据
    camera_calib = load_camera_calibration(camera_calib_path)

    # 为每个传感器创建校准信息
    sensor_token_to_channel = {sensor["token"]: sensor["channel"] for sensor in sensors}

    # 打印调试信息，确认映射关系
    print("相机通道与标定文件映射关系:")
    for sensor_token, channel in sensor_token_to_channel.items():
        if "CAM" in channel:
            # 相机校准 - 注意：根据实际视野位置映射，而非安装位置
            if "FRONT_LEFT" in channel:
                # NuScenes的FRONT_LEFT对应我们的rear_left（左后相机）
                calib_key = "rear_left"
                calib_file = "camera_6_left_back.json"
            elif "FRONT_RIGHT" in channel:
                # NuScenes的FRONT_RIGHT对应我们的rear_right（右后相机）
                calib_key = "rear_right"
                calib_file = "camera_7_right_back.json"
            elif "FRONT" in channel:
                calib_key = "front"
                calib_file = "camera_0_front100.json"
            elif "BACK_LEFT" in channel:
                # NuScenes的BACK_LEFT对应我们的front_left（左前相机）
                calib_key = "front_left"
                calib_file = "camera_3_left_front.json"
            elif "BACK_RIGHT" in channel:
                # NuScenes的BACK_RIGHT对应我们的front_right（右前相机）
                calib_key = "front_right"
                calib_file = "camera_4_right_front.json"
            elif "BACK" in channel:
                calib_key = "rear"
                calib_file = "camera_5_back.json"
            else:
                # 默认使用前相机
                calib_key = "front"
                calib_file = "camera_0_front100.json"

            print(f"{channel} -> {calib_key} -> {calib_file}")

            if calib_key in camera_calib:
                calib_data = camera_calib[calib_key]

                # 从外参矩阵中提取平移和旋转
                translation, rotation = extrinsic_matrix_to_translation_rotation(calib_data["extrinsic"])

                # 获取内参矩阵
                intrinsic = calib_data["intrinsic"]
                camera_intrinsic = [
                    [intrinsic[0], intrinsic[1], intrinsic[2]],
                    [intrinsic[3], intrinsic[4], intrinsic[5]],
                    [intrinsic[6], intrinsic[7], intrinsic[8]]
                ]
            else:
                # 如果没有找到标定数据，使用默认值
                translation = [1.7, 0.0, 1.5]
                rotation = [1.0, 0.0, 0.0, 0.0]
                camera_intrinsic = [
                    [1500.0, 0.0, 960.0],
                    [0.0, 1500.0, 600.0],
                    [0.0, 0.0, 1.0]
                ]
        else:
            # 激光雷达校准
            translation = [0.0, 0.0, 0.0]
            rotation = [1.0, 0.0, 0.0, 0.0]
            camera_intrinsic = []
        # 在创建校准传感器后调用检查
        # check_sensor_pose_reasonable(translation, rotation, channel)
        calibrated_sensor = {
            "token": str(uuid.uuid4()),
            "sensor_token": sensor_token,
            "translation": translation,
            "rotation": rotation,
            "camera_intrinsic": camera_intrinsic
        }

        calibrated_sensors.append(calibrated_sensor)

    return calibrated_sensors

def main():
    parser = argparse.ArgumentParser(description='Convert custom annotations to NuScenes format')
    parser.add_argument('--input', type=str, required=True, help='Path to input custom annotation JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory for NuScenes format files')

    args = parser.parse_args()

    # 加载自定义标注
    custom_annotations = load_custom_annotations(args.input)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 创建NuScenes类别
    categories = create_nuscenes_categories()

    # 获取所有自定义标签
    custom_labels = []
    for frame in custom_annotations:
        custom_labels.extend(frame.get("labels", []))

    # 创建自定义标签到NuScenes类别的映射
    custom_to_nuscenes = create_custom_to_nuscenes_mapping(custom_labels)

    # 创建NuScenes属性
    attributes = create_nuscenes_attributes()

    # 创建NuScenes可见性
    visibility = create_nuscenes_visibility()

    # 创建NuScenes传感器
    sensors = create_nuscenes_sensors()

    # 创建NuScenes校准传感器
    calibrated_sensors = create_nuscenes_calibrated_sensors(sensors)

    # 保存基本JSON文件
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

    print(f"Basic NuScenes JSON files have been saved to {args.output}")
    print("Now we need to create more complex files like scene, sample, sample_data, etc.")
    print("This will be handled by additional scripts.")

if __name__ == "__main__":
    main()
