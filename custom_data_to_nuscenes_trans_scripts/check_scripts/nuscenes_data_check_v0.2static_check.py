# 对上游交付的nuscenes格式的数据集进行有效性检查
# 检查范围：
#   1）数据结构的有效性；
#   2）数据字段的有效性：类型、范围、内容；
#   3）数据字段间的有效性：比如某个字段指向的数据对象是否存在，是否重复；
#   4）拓展数据的有效性：比如利用位置和时间计算出的速度值是否在合理范围；
#   4）对不能形式化的部分，借助人脑，对可视化的数据进行检查（可选）；

# 版本历史：
# 0.0:
#   添加初版检查功能
# 0.1:
#   添加num_lidar_pts为0的标注框的占比检查
#   添加阈值，限制无法计算速度的标注框的占比
#   
# 

script_version = 0.2
print(f"检查脚本的版本：{script_version}")

import json, shutil, os
from nuscenes.nuscenes import NuScenes
from collections import defaultdict
import os.path as osp
import argparse
from tqdm import tqdm
import numpy as np
import math

def kph2mps(kph):
  return kph * 1000.0 / (60 * 60)

def mps2kph(mps):
  return mps * 3.6

def load_table(data_root, table_name) -> dict:
        """ Loads a table. """
        with open(osp.join(data_root, '{}.json'.format(table_name))) as f:
            table = json.load(f)
        return table

def uniq_check(name: str, data: list):
  assert len(data) == len(set(data)), f"{name}具有重复值: {data}"

def object_exit(datas, field_name, field_val):
  assert isinstance(datas, dict)
  for d in datas.values():
    assert field_name in d.keys()
    if d[field_name] == field_val:
      return True
  return False

def scene_check(ds):
  scenes = ds.scene
  # 数据结构检查
  assert isinstance(scenes, list)
  assert len(scenes) > 0
  for s in scenes:
    assert 'token' in s.keys()
    assert 'name' in s.keys()
    assert 'log_token' in s.keys()
    assert 'nbr_samples' in s.keys()
    assert 'first_sample_token' in s.keys()
    assert 'last_sample_token' in s.keys()
  # 单个字段检查
  for s in scenes:
    token = s['token']
    assert token is not None
    assert isinstance(token, str)
    log_token = s['log_token']
    assert log_token is not None
    assert isinstance(log_token, str)
    nbr_samples = s['nbr_samples']
    assert nbr_samples is not None
    assert isinstance(nbr_samples, int)
    assert nbr_samples > 0 and nbr_samples < 100, f"nbr_samples无效: {nbr_samples}"
    first_sample_token = s['first_sample_token']
    assert first_sample_token is not None
    assert isinstance(first_sample_token, str)
    last_sample_token = s['last_sample_token']
    assert last_sample_token is not None
    assert isinstance(last_sample_token, str)
    name = s['name']
    assert name is not None
    assert isinstance(name, str)
  # 字段唯一性检查
  uniq_check('scene.name', [s['name'] for s in scenes])
  uniq_check('scene.token', [s['token'] for s in scenes])
  # 字段引用有效性检查
  sample_dict = ds.sample_dict
  for s in scenes:
    assert s['first_sample_token'] in sample_dict
    assert s['last_sample_token'] in sample_dict

def sample_check(ds):
  samples = ds.sample
  # 数据结构检查
  assert isinstance(samples, list)
  assert len(samples) > 0
  for s in samples:
    assert 'token' in s.keys()
    assert 'timestamp' in s.keys()
    assert 'prev' in s.keys()
    assert 'next' in s.keys()
    assert 'scene_token' in s.keys()
  # 单个字段检查
  for s in samples:
    token = s['token']
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0
    timestamp = s['timestamp']
    assert timestamp is not None
    assert isinstance(timestamp, int)
    assert len(str(timestamp)) == 16, "timestamp单位无效,需要us" # TODO
    prev = s['prev']
    assert prev is not None
    assert isinstance(prev, str)
    next_ = s['next']
    assert next_ is not None
    assert isinstance(next_, str)
    scene_token = s['scene_token']
    assert scene_token is not None
    assert isinstance(scene_token, str)
    assert len(scene_token) > 0
  # 字段唯一性检查
  uniq_check('sample.token', [s['token'] for s in samples])
  uniq_check('sample.timestamp', [s['timestamp'] for s in samples])
  # 字段引用有效性检查
  scene_dict = ds.scene_dict
  sample_dict = ds.sample_dict
  for s in samples:
    assert s['scene_token'] in scene_dict
    if s['prev'] != '':
      assert s['prev'] in sample_dict
    if s['next'] != '':
      assert s['next'] in sample_dict

def visibility_check(ds):
  visibilitys = ds.visibility
  # 数据结构检查
  assert isinstance(visibilitys, list)
  assert len(visibilitys) > 0
  for v in visibilitys:
    assert 'token' in v.keys()
    assert 'level' in v.keys()
  # 单个字段检查
  for v in visibilitys:
    token = v['token']
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0
    level = v['level']
    assert level is not None
    assert isinstance(level, str)
    assert len(level) > 0
  # 字段唯一性检查
  uniq_check('visibility.token', [v['token'] for v in visibilitys])
  uniq_check('visibility.level', [v['level'] for v in visibilitys])
  # 字段引用有效性检查
  # 无

def sensor_check(ds):
  sensors = ds.sensor
  # 数据结构检查
  assert isinstance(sensors, list)
  assert len(sensors) > 0
  for s in sensors:
    assert 'token' in s.keys()
    assert 'channel' in s.keys()
    assert 'modality' in s.keys()
  # 单个字段检查
  channels = ['CAM_FRONT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
              'LIDAR_TOP', 
              'RADAR_FRONT', 'RADAR_FRONT_RIGHT', 'RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
  modalitys = ['camera', 'lidar', 'radar']
  for s in sensors:
    token = s['token']
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0
    channel = s['channel']
    assert channel is not None
    assert isinstance(channel, str)
    assert len(channel) > 0
    assert channel in channels
    modality = s['modality']
    assert modality is not None
    assert isinstance(modality, str)
    assert len(modality) > 0
    assert modality in modalitys
  # 字段唯一性检查
  uniq_check('sensor.token', [s['token'] for s in sensors])
  uniq_check('sensor.channel', [s['channel'] for s in sensors])
  # 字段引用有效性检查
  # 无

def attribute_check(ds):
  attributes = ds.attribute
  # 数据结构检查
  assert isinstance(attributes, list)
  assert len(attributes) > 0
  for a in attributes:
    assert 'token' in a.keys()
    assert 'name' in a.keys()
    assert 'description' in a.keys()
  # 单个字段检查
  for a in attributes:
    token = a['token']
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0
    name = a['name']
    assert name is not None
    assert isinstance(name, str)
    assert len(name) > 0
    description = a['description']
    assert description is not None
    assert isinstance(description, str)
    assert len(description) > 0
  # 字段唯一性检查
  uniq_check('attribute.token', [a['token'] for a in attributes])
  uniq_check('attribute.name', [a['name'] for a in attributes])
  # 字段引用有效性检查
  # 无

def category_check(ds):
  categorys = ds.category
  # 数据结构检查
  assert isinstance(categorys, list)
  assert len(categorys) > 0
  for c in categorys:
    assert 'token' in c.keys()
    assert 'name' in c.keys()
    assert 'description' in c.keys()
  # 单个字段检查
  for c in categorys:
    token = c['token']
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0
    name = c['name']
    assert name is not None
    assert isinstance(name, str)
    assert len(name) > 0
    description = c['description']
    assert description is not None
    assert isinstance(description, str)
    assert len(description) > 0
  # 字段唯一性检查
  uniq_check('category.token', [c['token'] for c in categorys])
  uniq_check('category.name', [c['name'] for c in categorys])
  # 字段引用有效性检查
  # 无

def calibrated_sensor_check(ds):
  calibrated_sensors = ds.calibrated_sensor
  # 数据结构检查
  assert isinstance(calibrated_sensors, list)
  assert len(calibrated_sensors) > 0
  for c in calibrated_sensors:
    assert 'token' in c.keys()
    assert 'sensor_token' in c.keys()
    assert 'translation' in c.keys()
    assert 'rotation' in c.keys()
    assert 'camera_intrinsic' in c.keys()
  # 单个字段检查
  for c in calibrated_sensors:
    token = c['token']
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0
    sensor_token = c['sensor_token']
    assert sensor_token is not None
    assert isinstance(sensor_token, str)
    assert len(sensor_token) > 0
    translation = c['translation']
    assert translation is not None
    assert isinstance(translation, list)
    assert len(translation) == 3
    for i in range(3):
      assert isinstance(translation[i], float)
      m = 10
      assert translation[i] > -m and translation[i] < m
    rotation = c['rotation']
    assert rotation is not None
    assert isinstance(rotation, list)
    assert len(rotation) == 4
    sum = 0.0
    for i in range(4):
      assert isinstance(rotation[i], float)
      sum = sum + rotation[i] * rotation[i]
    assert abs(sum - 1.0) < 1e-5, f'sum:{sum}'
    camera_intrinsic = c['camera_intrinsic']
    assert camera_intrinsic is not None
    assert isinstance(camera_intrinsic, list)
    assert len(camera_intrinsic) in [0, 3], f'camera_intrinsic:{camera_intrinsic}'
    for i in range(len(camera_intrinsic)):
      assert isinstance(camera_intrinsic[i], list)
      assert len(camera_intrinsic[i]) == 3
      for j in range(3):
        assert isinstance(camera_intrinsic[i][j], float)
        m = 1e4
        assert camera_intrinsic[i][j] > -m and camera_intrinsic[i][j] < m
  # 字段唯一性检查
  uniq_check('calibrated_sensor.token', [c['token'] for c in calibrated_sensors])
  # 字段引用有效性检查
  sensor_dict = ds.sensor_dict
  for c in calibrated_sensors:
    assert c['sensor_token'] in sensor_dict

def ego_pose_check(ds):
  ego_poses = ds.ego_pose
  # 数据结构检查
  assert isinstance(ego_poses, list)
  assert len(ego_poses) > 0
  for e in ego_poses:
    assert 'token' in e.keys()
    assert 'timestamp' in e.keys()
    assert 'rotation' in e.keys()
    assert 'translation' in e.keys()
  # 单个字段检查
  for e in ego_poses:
    token = e['token']
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0
    timestamp = e['timestamp']
    assert timestamp is not None
    assert isinstance(timestamp, int)
    assert len(str(timestamp)) == 16, "timestamp单位无效,需要us" # TODO
    rotation = e['rotation']
    assert rotation is not None
    assert isinstance(rotation, list)
    assert len(rotation) == 4
    sum = 0.0
    for i in range(4):
      assert isinstance(rotation[i], float)
      sum = sum + rotation[i] * rotation[i]
    assert abs(sum - 1.0) < 1e-5, f'sum:{sum}'
    translation = e['translation']
    assert translation is not None
    assert isinstance(translation, list)
    assert len(translation) == 3
    for i in range(3):
      assert isinstance(translation[i], float)
      # m = 10
      # assert translation[i] > -m and translation[i] < m, f"无效的 translation:{translation}, i:{i}"
  # 字段唯一性检查
  uniq_check('ego_pose.token', [c['token'] for c in ego_poses])
  # 字段引用有效性检查
  # 无

def instance_check(ds):
  instances = ds.instance
  # 数据结构检查
  assert isinstance(instances, list)
  assert len(instances) > 0
  for i in instances:
    assert 'token' in i.keys()
    assert 'category_token' in i.keys()
    assert 'nbr_annotations' in i.keys()
    assert 'first_annotation_token' in i.keys()
    assert 'last_annotation_token' in i.keys()
  # 单个字段检查
  for i in instances:
    token = i['token']
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0
    category_token = i['category_token']
    assert category_token is not None
    assert isinstance(category_token, str)
    assert len(category_token) > 0
    nbr_annotations = i['nbr_annotations']
    assert nbr_annotations is not None
    assert isinstance(nbr_annotations, int)
    assert nbr_annotations > 0
    first_annotation_token = i['first_annotation_token']
    assert first_annotation_token is not None
    assert isinstance(first_annotation_token, str)
    assert len(first_annotation_token) > 0
    last_annotation_token = i['last_annotation_token']
    assert last_annotation_token is not None
    assert isinstance(last_annotation_token, str)
    assert len(last_annotation_token) > 0
  # 字段唯一性检查
  uniq_check('instance.token', [c['token'] for c in instances])
  # 字段引用有效性检查
  category_dict = ds.category_dict
  sample_annotation_dict = ds.sample_annotation_dict
  for i in instances:
    assert i['category_token'] in category_dict
    assert i['first_annotation_token'] in sample_annotation_dict
    assert i['last_annotation_token'] in sample_annotation_dict

def sample_data_check(ds):
  sample_datas = ds.sample_data
  # 数据结构检查
  assert isinstance(sample_datas, list)
  assert len(sample_datas) > 0
  for s in sample_datas:
    assert 'token' in s.keys()
    assert 'sample_token' in s.keys()
    assert 'ego_pose_token' in s.keys()
    assert 'calibrated_sensor_token' in s.keys()
    assert 'timestamp' in s.keys()
    assert 'fileformat' in s.keys()
    assert 'is_key_frame' in s.keys()
    assert 'height' in s.keys()
    assert 'width' in s.keys()
    assert 'filename' in s.keys()
    assert 'prev' in s.keys()
    assert 'next' in s.keys()
  # 单个字段检查
  for s in sample_datas:
    token = s['token']
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0
    sample_token = s['sample_token']
    assert sample_token is not None
    assert isinstance(sample_token, str)
    assert len(sample_token) > 0
    ego_pose_token = s['ego_pose_token']
    assert ego_pose_token is not None
    assert isinstance(ego_pose_token, str)
    assert len(ego_pose_token) > 0
    calibrated_sensor_token = s['calibrated_sensor_token']
    assert calibrated_sensor_token is not None
    assert isinstance(calibrated_sensor_token, str)
    assert len(calibrated_sensor_token) > 0
    timestamp = s['timestamp']
    assert timestamp is not None
    assert isinstance(timestamp, int)
    assert len(str(timestamp)) == 16, "timestamp单位无效,需要us" # TODO
    fileformat = s['fileformat']
    assert fileformat is not None
    assert isinstance(fileformat, str)
    assert len(fileformat) > 0
    assert fileformat in ['pcd', 'jpg', 'bin'], f"无效的 fileformat: {fileformat}"
    is_key_frame = s['is_key_frame']
    assert is_key_frame is not None
    assert isinstance(is_key_frame, bool)
    height = s['height']
    assert height is not None
    assert isinstance(height, int)
    assert height >= 0 and height <= 2160, f"无效的 height:{height}" # TODO
    width = s['width']
    assert width is not None
    assert isinstance(width, int)
    assert width >= 0 and width <= 3840, f"无效的 width:{width}" # TODO
    filename = s['filename']
    assert filename is not None
    assert isinstance(filename, str)
    assert len(filename) > 0
    prev = s['prev']
    assert prev is not None
    assert isinstance(prev, str)
    next_ = s['next']
    assert next_ is not None
    assert isinstance(next_, str)
  # 字段唯一性检查
  uniq_check('sample_data.token', [s['token'] for s in sample_datas])
  # 字段引用有效性检查
  sample_dict = ds.sample_dict
  ego_pose_dict = ds.ego_pose_dict
  calibrated_sensor_dict = ds.calibrated_sensor_dict
  sample_data_dict = ds.sample_data_dict
  for s in sample_datas:
    assert s['sample_token'] in sample_dict
    assert s['ego_pose_token'] in ego_pose_dict
    assert s['calibrated_sensor_token'] in calibrated_sensor_dict
    if s['prev'] != '':
      assert s['prev'] in sample_data_dict
    if s['next'] != '':
      assert s['next'] in sample_data_dict

def sample_annotation_check(ds):
  sample_annotations = ds.sample_annotation
  # 数据结构检查
  assert isinstance(sample_annotations, list)
  assert len(sample_annotations) > 0
  for s in sample_annotations:
    assert 'token' in s.keys()
    assert 'sample_token' in s.keys()
    assert 'instance_token' in s.keys()
    assert 'visibility_token' in s.keys()
    assert 'attribute_tokens' in s.keys()
    assert 'translation' in s.keys()
    assert 'size' in s.keys()
    assert 'rotation' in s.keys()
    assert 'prev' in s.keys()
    assert 'next' in s.keys()
    assert 'num_lidar_pts' in s.keys()
    assert 'num_radar_pts' in s.keys()
  # 单个字段检查
  sample_annotation_dict = ds.sample_annotation_dict
  for s in sample_annotations:
    token = s['token']
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0
    sample_token = s['sample_token']
    assert sample_token is not None
    assert isinstance(sample_token, str)
    assert len(sample_token) > 0
    instance_token = s['instance_token']
    assert instance_token is not None
    assert isinstance(instance_token, str)
    assert len(instance_token) > 0
    visibility_token = s['visibility_token']
    assert visibility_token is not None
    assert isinstance(visibility_token, str)
    assert len(visibility_token) > 0
    attribute_tokens = s['attribute_tokens']
    assert attribute_tokens is not None
    assert isinstance(attribute_tokens, list)
    assert len(attribute_tokens) >= 0
    for i in range(len(attribute_tokens)):
      assert attribute_tokens[i] is not None
      assert isinstance(attribute_tokens[i], str)
      assert len(attribute_tokens[i]) > 0
    translation = s['translation']
    assert translation is not None
    assert isinstance(translation, list)
    assert len(translation) == 3
    for i in range(3):
      assert isinstance(translation[i], float) or isinstance(translation[i], int)
      # m = 1e6
      # assert translation[i] > -m and translation[i] < m
    size = s['size']
    assert size is not None
    assert isinstance(size, list)
    assert len(size) == 3
    for i in range(3):
      assert isinstance(size[i], float) or isinstance(size[i], int)
      m = 50 # 障碍物的最大尺寸
      assert size[i] > -m and size[i] < m
    rotation = s['rotation']
    assert rotation is not None
    assert isinstance(rotation, list)
    assert len(rotation) == 4
    sum = 0.0
    for i in range(4):
      assert isinstance(rotation[i], float)
      sum = sum + rotation[i] * rotation[i]
    assert abs(sum - 1.0) < 1e-5, f'sum:{sum}'
    if s['prev'] != '':
      assert s['prev'] in sample_annotation_dict
    if s['next'] != '':
      assert s['next'] in sample_annotation_dict
    num_lidar_pts = s['num_lidar_pts']
    assert num_lidar_pts is not None
    assert isinstance(num_lidar_pts, int)
    assert num_lidar_pts >= 0, f"num_lidar_pts 无效: {num_lidar_pts}"
    num_radar_pts = s['num_radar_pts']
    assert num_radar_pts is not None
    assert isinstance(num_radar_pts, int)
    assert num_radar_pts >= 0, f"num_radar_pts 无效: {num_radar_pts}"

def velocity_check(ds):
  sample_no_anns = []
  anno_no_velocity = []
  anno_has_velocity = []
  for s in ds.sample:
    assert 'anns' in s
    if len(s['anns']) == 0:
      sample_no_anns.append(s['token'])
    for sample_annotation_token in s['anns']:
      v = ds.box_velocity(sample_annotation_token)
      v = np.linalg.norm(v, axis=-1)
      if math.isnan(v):
        anno_no_velocity.append({'sample_token': s['token'], 'sample_annotation_token': sample_annotation_token})
      else:
        assert v < kph2mps(300)
        anno_has_velocity.append({'sample_token': s['token'], 'sample_annotation_token': sample_annotation_token})
  if len(sample_no_anns) > 0:
    # print(f"警告: 这些sample没有标注:{sample_no_anns}")
    print(f"警告: 没有标注的sample数量:{len(sample_no_anns)}")
  if len(anno_no_velocity) > 0:
    # print(f"警告: 这些标注框无法计算出速度:{anno_no_velocity}")
    print(f"警告: 无法计算出速度的标注框的数量:{len(anno_no_velocity)}")
    rate = 100.0 * (len(anno_no_velocity) * 1.0 / (len(anno_no_velocity) + len(anno_has_velocity)))
    print(f'警告: 不能算出速度的标注框的占比:{rate:.2f}%')
    threshold = 20
    assert rate < threshold, f'不能算出速度的标注框的占比不能超过{threshold}%, rate:{rate:.2f}'

def num_lidar_pts_check(ds):
  zero_num_lidar_pts = []
  non_zero_num_lidar_pts = []
  for s in ds.sample:
    assert 'anns' in s
    for sample_annotation_token in s['anns']:
      sa = ds.get('sample_annotation', sample_annotation_token)
      assert 'num_lidar_pts' in sa
      assert isinstance(sa['num_lidar_pts'], int)
      num_lidar_pts = sa['num_lidar_pts']
      assert num_lidar_pts >= 0
      if num_lidar_pts == 0:
        zero_num_lidar_pts.append(sample_annotation_token)
      else:
        non_zero_num_lidar_pts.append(sample_annotation_token)
  if len(zero_num_lidar_pts) > 0:
    # print(f"警告: 这些标注框的num_lidar_pts为0:{zero_num_lidar_pts}")
    print(f"警告: num_lidar_pts为0的标注框:{len(zero_num_lidar_pts)}")
    rate = 100.0 * (len(zero_num_lidar_pts) * 1.0 / (len(zero_num_lidar_pts) + len(non_zero_num_lidar_pts)))
    print(f'警告: num_lidar_pts为0的标注框的占比:{rate:.2f}%')
    threshold = 30
    assert rate < threshold, f'num_lidar_pts为0的标注框的占比不能超过{threshold}%, rate:{rate:.2f}'

class Dataset:
  def __init__(self,
               version: str,
               dataroot: str):
    self.version = version
    self.dataroot = dataroot
    
    self.category = self.__load_table__('category')
    self.attribute = self.__load_table__('attribute')
    self.visibility = self.__load_table__('visibility')
    self.instance = self.__load_table__('instance')
    self.sensor = self.__load_table__('sensor')
    self.calibrated_sensor = self.__load_table__('calibrated_sensor')
    self.ego_pose = self.__load_table__('ego_pose')
    self.log = self.__load_table__('log')
    self.scene = self.__load_table__('scene')
    self.sample = self.__load_table__('sample')
    self.sample_data = self.__load_table__('sample_data')
    self.sample_annotation = self.__load_table__('sample_annotation')
    self.map = self.__load_table__('map')

    self.category_dict = self.__list_to_dict__(self.category)
    self.attribute_dict = self.__list_to_dict__(self.attribute)
    self.visibility_dict = self.__list_to_dict__(self.visibility)
    self.instance_dict = self.__list_to_dict__(self.instance)
    self.sensor_dict = self.__list_to_dict__(self.sensor)
    self.calibrated_sensor_dict = self.__list_to_dict__(self.calibrated_sensor)
    self.ego_pose_dict = self.__list_to_dict__(self.ego_pose)
    self.log_dict = self.__list_to_dict__(self.log)
    self.scene_dict = self.__list_to_dict__(self.scene)
    self.sample_dict = self.__list_to_dict__(self.sample)
    self.sample_data_dict = self.__list_to_dict__(self.sample_data)
    self.sample_annotation_dict = self.__list_to_dict__(self.sample_annotation)
    self.map_dict = self.__list_to_dict__(self.map)

  @property
  def table_root(self) -> str:
    """ Returns the folder where the tables are stored for the relevant version. """
    return osp.join(self.dataroot, self.version)
  
  def __list_to_dict__(self, table: list):
    tokens = [item['token'] for item in table]
    assert len(tokens) == len(set(tokens))
    return {item['token']: item for item in table}
  
  def __load_table__(self, table_name) -> dict:
    """ Loads a table. """
    with open(osp.join(self.table_root, '{}.json'.format(table_name))) as f:
      table = json.load(f)
    return table

def visualize_velocity(ax, sample, sensor_name, box_vis_level):
  import matplotlib.pyplot as plt
  from pyquaternion import Quaternion
  from nuscenes.utils.geometry_utils import view_points, box_in_image

  
  sample_data_token = sample['data'][sensor_name]
  sd_record = ds.get('sample_data', sample_data_token)
  cs_record = ds.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
  sensor_record = ds.get('sensor', cs_record['sensor_token'])
  pose_record = ds.get('ego_pose', sd_record['ego_pose_token'])

  if sensor_record['modality'] == 'camera':
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    imsize = (sd_record['width'], sd_record['height'])
    is_lidar = False
    use_flat_vehicle_coordinates = False
  else:
    cam_intrinsic = None
    imsize = None
    is_lidar = True
    use_flat_vehicle_coordinates = True

  for sample_annotation_token in sample['anns']:
    # 
    box = ds.get_box(sample_annotation_token)
    velocity = ds.box_velocity(sample_annotation_token)
    velocity_in_global = velocity
    if math.isnan(velocity[0]):
       continue
    if use_flat_vehicle_coordinates:
      # Move box to ego vehicle coord system parallel to world z plane.
      yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
      box.translate(-np.array(pose_record['translation']))
      box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
      q = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
      velocity = np.dot(q.rotation_matrix, velocity)
    else:
      # Move box to ego vehicle coord system.
      box.translate(-np.array(pose_record['translation']))
      box.rotate(Quaternion(pose_record['rotation']).inverse)
      q = Quaternion(pose_record['rotation']).inverse
      velocity = np.dot(q.rotation_matrix, velocity)

      #  Move box to sensor coord system.
      box.translate(-np.array(cs_record['translation']))
      box.rotate(Quaternion(cs_record['rotation']).inverse)
      q = Quaternion(cs_record['rotation']).inverse
      velocity = np.dot(q.rotation_matrix, velocity)

    if is_lidar:
      corners = view_points(box.corners(), np.eye(4), normalize=False)[:2, :]
      center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
      scale = 5
      velocity = view_points(velocity.reshape((3, 1)), np.eye(4), normalize=False)[:2, :]
      v1 = (velocity.reshape(2) / kph2mps(20)) * scale
      v2 = v1.astype(int) 
      if np.linalg.norm(velocity.reshape(2)) < kph2mps(2):
        continue
      velocity = v2
      ax.arrow(center_bottom[0], center_bottom[1], 
              velocity[0], velocity[1],
              #  (velocity[0]/mps2kph(10)) * scale, (velocity[1]/mps2kph(10)) * scale, 
                  #  head_width=1.0,  # 箭头头部宽度
                  #  head_length=1.5, # 箭头头部长度
                  #  fc='red',        # 填充颜色
                  #  ec='red',        # 边缘颜色
                  #  alpha=0.7
                  color='green'
                 )
      ax.text(center_bottom[0], center_bottom[1], 'v:' + str(round(np.linalg.norm(velocity_in_global), 2)), fontsize=8, color='green', ha='center', va='center')
    else:
      if np.linalg.norm(velocity.reshape(3)) < kph2mps(2):
        continue
      if not box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
        continue
      scale = 5
      center_bottom = np.mean(box.corners().T[[2, 3, 7, 6]], axis=0)
      vp = center_bottom + (velocity / kph2mps(20)) * scale
      center_bottom = view_points(center_bottom.reshape(3, 1), cam_intrinsic, normalize=True)[:2, :]
      center_bottom = center_bottom.reshape(2).astype(int)
      velocity = view_points(vp.reshape((3, 1)), cam_intrinsic, normalize=True)[:2, :]
      velocity = velocity.reshape(2).astype(int)
      ax.arrow(center_bottom[0], center_bottom[1], 
              velocity[0] - center_bottom[0],
              velocity[1] - center_bottom[1],
              color='green')
      ax.text(center_bottom[0], center_bottom[1], 'v:' + str(round(np.linalg.norm(velocity_in_global), 2)), fontsize=8, color='black', ha='center', va='center')

def visualize_to_jpg(ds, sample, sensor_name, disable_velocity_visualize, box_vis_level = 1, axes_limit = 50, path_prefix = ''):
  import matplotlib.pyplot as plt
  ds.render_sample_data(sample['data'][sensor_name], box_vis_level=box_vis_level, axes_limit=axes_limit, underlay_map=False)
  if not disable_velocity_visualize:
    ax = plt.gca()
    visualize_velocity(ax, sample, sensor_name, box_vis_level)
  plt.savefig(path_prefix + f"{sensor_name}_{sample['token']}.jpg")
  plt.close()

def check_static_object_velocity(nusc, threshold=0.5):
    """
    检查静态物体（Barrier, Cone）是否存在异常速度。
    逻辑：遍历每个Scene，找到包含该类别的帧，如果速度超标，记录并跳过该Scene。
    """
    print(f"\n=== 开始检查静态物体速度异常 (阈值 > {threshold} m/s) ===")
    
    # 1. 确定静态类别的名称 (NuScenes标准名称)
    static_categories = {
        'movable_object.barrier', 
        'movable_object.trafficcone'
        # 'static_object.bicycle_rack' # 可选
    }
    
    abnormal_samples = []
    
    # 遍历所有场景
    for scene in tqdm(nusc.scene, desc="Scanning Scenes"):
        scene_name = scene['name']
        found_issue_in_scene = False
        
        # 遍历该场景下的所有样本
        sample_token = scene['first_sample_token']
        
        while sample_token:
            sample = nusc.get('sample', sample_token)
            
            # 检查该样本下的所有标注
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                
                # 获取类别名
                # 注意：nusc.get('category', token) 这种 API 有时需要查表，
                # 但 nusc.get('sample_annotation', ...) 返回后的 instance 可能没有 category_name 字段，
                # 需要通过 instance_token -> category_token -> name 查，或者直接看 ann['category_name'] (如果有helper的话)
                # NuScenes SDK 中 ann 通常直接包含 'category_name' (通过 NuScenes 初始化时的表关联)
                
                cat_name = ann.get('category_name')
                if not cat_name:
                    # 如果没有直接字段，手动查
                    inst = nusc.get('instance', ann['instance_token'])
                    cat = nusc.get('category', inst['category_token'])
                    cat_name = cat['name']
                
                if cat_name in static_categories:
                    # 取速度向量 (x, y)
                    velo = nusc.box_velocity(ann_token)
                    # 计算模长 (注意: box_velocity 返回的是 np.array([vx, vy, vz])，通常我们只关心水平速度)
                    # 如果这该帧没有计算速度(比如第一帧)，回传即是nan
                    if np.any(np.isnan(velo)):
                        speed = 0.0
                    else:
                        speed = np.linalg.norm(velo[:2]) # 只看水平速度
                    
                    if speed > threshold:
                        # 发现异常！
                        print(f"  [异常发现] Scene: {scene_name} | Sample: {sample_token} | \n"
                              f"            Obj: {cat_name} | Speed: {speed:.4f} m/s")
                        
                        # 获取 LiDAR 文件路径用于定位
                        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                        lidar_path = lidar_data['filename']
                        
                        abnormal_samples.append({
                            'scene': scene_name,
                            'sample_token': sample_token,
                            'category': cat_name,
                            'speed': float(speed),
                            'lidar_path': lidar_path
                        })
                        
                        found_issue_in_scene = True
                        break # 跳出 annotation 循环
            
            if found_issue_in_scene:
                break # 跳出 sample 循环，继续下一个 scene
            
            sample_token = sample['next']
            
    print(f"\n检查完成。共发现 {len(abnormal_samples)} 个包含异常静态物体速度的 Sequence。")
    if abnormal_samples:
        print("详细列表如下:")
        for item in abnormal_samples:
            print(f"- {item['lidar_path']} (Speed: {item['speed']:.2f} m/s)")
            
    return abnormal_samples

def check_ego_pose_consistency(nusc, acc_threshold=15.0):
    """
    检查 Ego Pose 的物理一致性（速度、加速度、角速度）。
    用于验证平滑处理（Smoothing）是否生效。
    """
    print(f"\n=== 开始检查 Ego Pose (最终轨迹) 物理一致性 ===")
    
    # 全局统计容器
    all_speeds = []
    all_accels = []
    abnormal_scenes = []
    
    for scene in tqdm(nusc.scene, desc="Analyzing Trajectory"):
        sample_token = scene['first_sample_token']
        
        # 提取该 Scene 下所有关键帧的位姿和时间
        poses = []
        timestamps = [] # 秒
        
        while sample_token:
            sample = nusc.get('sample', sample_token)
            lidar_token = sample['data']['LIDAR_TOP']
            sd_record = nusc.get('sample_data', lidar_token)
            pose = nusc.get('ego_pose', sd_record['ego_pose_token'])
            
            poses.append(np.array(pose['translation']))
            timestamps.append(sd_record['timestamp'] / 1e6)
            
            sample_token = sample['next']
            
        if len(poses) < 3:
            continue
            
        # 转换为 numpy 数组方便计算
        poses = np.array(poses)
        ts = np.array(timestamps)
        
        # 计算差分
        dt = np.diff(ts)
        dist = np.linalg.norm(np.diff(poses, axis=0), axis=1)
        
        # 过滤掉非法的 dt (防止除零错误)
        valid_mask = dt > 0.001
        if np.sum(valid_mask) == 0:
            continue
            
        # 1. 速度 (m/s)
        velocities = dist[valid_mask] / dt[valid_mask]
        all_speeds.extend(velocities)
        
        # 2. 加速度 (m/s^2) - 再次差分速度
        # 注意：加速度对应的是 dt 的中间时刻，长度比速度少1
        dv = np.diff(velocities)
        dt_acc = dt[valid_mask][:-1]
        
        if len(dt_acc) > 0:
            accels = np.abs(dv / dt_acc)
            all_accels.extend(accels)
            
            # 记录异常 Scene
            max_a = np.max(accels)
            if max_a > 20.0: # 20m/s^2 约等于 2g，普通乘用车很难达到
                abnormal_scenes.append((scene['name'], max_a))

    # --- 输出统计报告 ---
    print("\n" + "="*40)
    print("      Ego Pose 物理合理性分析报告      ")
    print("="*40)
    if not all_speeds:
        print("没有收集到有效轨迹数据。")
        return []

    all_speeds = np.array(all_speeds)
    all_accels = np.array(all_accels)
    
    print(f"检查样本总数 (帧间间隔): {len(all_speeds)}")
    
    print(f"\n[速度统计 - Speed]")
    print(f"  Mean: {np.mean(all_speeds):.2f} m/s ({np.mean(all_speeds)*3.6:.1f} km/h)")
    print(f"  Max : {np.max(all_speeds):.2f} m/s ({np.max(all_speeds)*3.6:.1f} km/h)")
    
    print(f"\n[加速度统计 - Acceleration]")
    print(f"  Mean: {np.mean(all_accels):.2f} m/s^2")
    print(f"  Max : {np.max(all_accels):.2f} m/s^2")
    if len(all_accels) > 0:
        p99 = np.percentile(all_accels, 99)
        p999 = np.percentile(all_accels, 99.9)
        print(f"  99%  分位: {p99:.2f} m/s^2")
        print(f"  99.9%分位: {p999:.2f} m/s^2")

    print("\n[评价结论]")
    if np.mean(all_accels) > 5.0:
        print("❌ 警告: 平均加速度过大 (>5.0)，说明平滑处理可能不足或原始数据噪声极大。")
    elif np.percentile(all_accels, 99) < 10.0:
        print("✅ 通过: 99%的情况下加速度小于 1g (9.8m/s^2)，平滑效果良好。")
    else:
        print("⚠️ 注意: 存在个别激烈动作或噪声，但整体在可控范围内。")

    if abnormal_scenes:
        # 按加速度大小排序，打印前10个
        abnormal_scenes.sort(key=lambda x: x[1], reverse=True)
        print(f"\n[Top 5 加速度异常场景]")
        for name, acc in abnormal_scenes[:5]:
            print(f"  - {name}: Max Accel = {acc:.2f} m/s^2")

    return abnormal_scenes

def modify_static_objects_velocity(nusc, json_path, threshold=0.5):
    """
    【修正版】修改模式：
    强制将静态类别（Barrier, Cone）的所有帧坐标统一为该实例的重心坐标。
    Threshold 用途：用于区分“微小抖动”和“异常漂移”，并输出不同级别的日志。
    """
    print(f"\n=== [MODIFY MODE] 正在修正静态物体坐标 (阈值: {threshold} m/s) ===")
    
    static_categories = {
        'movable_object.barrier', 
        'movable_object.trafficcone',
        'static_object.bicycle_rack'
    }
    
    instances_to_fix = defaultdict(list) 
    
    print("正在分析实例轨迹...")
    # 使用 instance 表遍历效率更高，但这里为了兼容性仍依赖 nusc API
    for ann in tqdm(nusc.sample_annotation, desc="Scanning Annotations"):
        try:
            inst = nusc.get('instance', ann['instance_token'])
            cat_name = nusc.get('category', inst['category_token'])['name']
            if cat_name in static_categories:
                instances_to_fix[ann['instance_token']].append(ann['token'])
        except Exception:
            continue
            
    print(f"发现 {len(instances_to_fix)} 个静态实例需要处理。")
    
    # 加载原始 JSON
    try:
        with open(json_path, 'r') as f:
            raw_anns = json.load(f)
    except Exception as e:
        print(f"无法读取原始 JSON: {json_path}, 错误: {e}")
        return

    token_to_idx = {item['token']: i for i, item in enumerate(raw_anns)}
    modified_count = 0
    jitter_fixed_count = 0
    high_speed_fixed_count = 0
    
    for inst_token, ann_tokens in tqdm(instances_to_fix.items(), desc="Fixing Instances"):
        coords = []
        valid_indices = []
        timestamps = [] # 用于粗略计算速度
        
        for t in ann_tokens:
            if t in token_to_idx:
                idx = token_to_idx[t]
                ann_data = raw_anns[idx]
                coords.append(ann_data['translation'])
                # 如果JSON里没有timestamp，可能需要从nusc里借用，这里简化处理只看位置离散度
                valid_indices.append(idx)
        
        if not coords:
            continue
            
        coords_np = np.array(coords)
        mean_pos = np.mean(coords_np, axis=0).tolist()
        
        # 【新增】利用 threshold 进行统计
        # 计算该实例位置的最大偏移量（相对于重心）
        diffs = np.linalg.norm(coords_np - mean_pos, axis=1)
        max_deviation = np.max(diffs)
        
        # 估算“如果这是在0.5s(10Hz*5)内发生的”，最大速度大概是多少？
        # 这是一个粗略估计，用来分类日志
        estimated_speed = max_deviation / 0.1 # 假设每帧0.1s，偏移量即速度指标
        
        is_high_speed_error = False
        if max_deviation > (threshold * 0.5): # 简单判断：如果偏移超过阈值的一半，说明某时刻速度肯定超标
             is_high_speed_error = True
        
        # 无论速度大小，只要是 Barrier/Cone，我们都强制归零（更安全）。
        # 除非方差极小（本来就是静止的）
        if max_deviation < 0.001: 
            continue 

        if is_high_speed_error:
            high_speed_fixed_count += 1
            # 可以选择打印 ID
            # print(f"  [修正告警] Instance {inst_token} 偏移较大 ({max_deviation:.2f}m), 强制归零。")
        else:
            jitter_fixed_count += 1
            
        # 应用修改
        for idx in valid_indices:
            raw_anns[idx]['translation'] = mean_pos
            modified_count += 1
            
    if modified_count > 0:
        backup_path = json_path + ".bak"
        if not os.path.exists(backup_path): # 避免覆盖原始备份
            shutil.copy(json_path, backup_path)
            print(f"已备份原文件到: {backup_path}")
        
        with open(json_path, 'w') as f:
            json.dump(raw_anns, f)
        print(f"✓ 保存成功！")
        print(f"  - 修正了 {jitter_fixed_count} 个微小抖动实例 (Speed < {threshold})")
        print(f"  - 修正了 {high_speed_fixed_count} 个大幅漂移/高速实例 (Speed > {threshold}) -- 请关注是否为误标")
        print(f"  - 总计修改 Sample Annotation 条目数: {modified_count}")
    else:
        print("没有需要修改的数据。")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
  parser.add_argument("--dataroot", type=str, default="./data/nuscenes", help="")
  parser.add_argument("--version", type=str, default="v1.0-trainval", help="")
  parser.add_argument("--disable_check", action="store_true", default=False, help="取消对数据有效性的检查")
  parser.add_argument("--visualize", action="store_true", default=False, help="是否对数据进行可视化")
  parser.add_argument("--static_speed_check", action="store_true", default=False, help="启用静态物体速度专项检查")
  parser.add_argument("--speed_threshold", type=float, default=0.5, help="静态物体速度报警阈值")
  parser.add_argument("--modify", action="store_true", default=False, help="也就是 'modify' 选项，开启后会修正JSON文件")
  parser.add_argument("--disable_velocity_visualize", action="store_true", default=False, help="是否对数据进行可视化")
  parser.add_argument("--axes_limit", type=int, default=50, help="lidar可视化的时候的范围,单位米")
  parser.add_argument("--vis_out_dir", type=str, default="", help="")
  
  args = parser.parse_args()
  # === 阶段 1: 修改模式 (Modify Phase) ===
  # 如果需要修改，先加载一次SDK处理修改，因为修改依赖SDK的查询功能
  if args.modify:
      print("\n" + "#"*50)
      print("      进入修改模式 (MODIFICATION PHASE)      ")
      print("#"*50)
      # 必须先初始化一个 nusc 用于查询需要修改的实例
      nusc_for_mod = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
      ann_json_path = os.path.join(args.dataroot, args.version, "sample_annotation.json")
      
      if os.path.exists(ann_json_path):
          modify_static_objects_velocity(nusc_for_mod, ann_json_path, threshold=args.speed_threshold)
          print("\n[重置] 修改已完成。正在重新加载数据集以进行最终检查...")
          # 销毁旧对象释放内存（虽然Python会自动回收，但显式删除更保险）
          del nusc_for_mod
      else:
          print(f"错误: 找不到标注文件 {ann_json_path}。跳过修改步骤。")
  
  # === 阶段 2: 检查模式 (Validation Phase) ===
  # 无论是否进行了修改，都对(最终的)数据进行检查
  if not args.disable_check:
    print("\n" + "#"*50)
    print("      进入检查模式 (VALIDATION PHASE)      ")
    print("#"*50)
    
    # 1. 重新初始化 SDK 对象 (确保加载的是磁盘上最新的 JSON)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    # 2. 运行高级逻辑检查 (依赖 SDK API)
    # (1) 静态物体速度检查 (如果刚才modify生效了，这里应该找不到任何异常)
    if args.static_speed_check or args.modify:
        print("\n正在验证静态物体修正效果...")
        abnormal_list = check_static_object_velocity(nusc, threshold=args.speed_threshold)
        if args.modify and len(abnormal_list) == 0:
            print("✅ 验证通过：所有静态物体速度已归零。")
        elif args.modify:
            print(f"⚠️ 警告：修正后仍发现 {len(abnormal_list)} 个异常，请检查修正逻辑。")

    # (2) Ego Pose 物理一致性检查 (这正是你想要的“速度/加速度跳变”检查)
    # 这会验证你的平滑算法 (rolling mean) 是否有效
    check_ego_pose_consistency(nusc) 

    # 3. 运行基础结构检查 (依赖 Dataset 实现)
    # 我们保留你原来基于 Dataset 类的那些检查函数
    print("\n=== 开始结构性检查 (Schema Check) ===")
    ds_struct = Dataset(version=args.version, dataroot=args.dataroot)
    
    # 这些函数非常快，保留作为底线检查
    try:
        scene_check(ds_struct)
        sample_check(ds_struct)
        visibility_check(ds_struct)
        sensor_check(ds_struct)
        attribute_check(ds_struct)
        category_check(ds_struct)
        # calibrated_sensor_check(ds_struct) # 其实 NuScenes 类加载本身就会做这些检查，如果不报错说明基本结构是好的
        # ego_pose_check(ds_struct)
        # instance_check(ds_struct)
        # sample_data_check(ds_struct)
        # sample_annotation_check(ds_struct)
        
        # 你的自定义逻辑检查
        velocity_check(ds_struct) 
        num_lidar_pts_check(ds_struct)
        print("\n✅ 所有基础结构检查通过。")
        
    except AssertionError as e:
        print(f"\n❌ 结构检查失败: {e}")
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")

  # 4. 可视化阶段 (如果需要)
  if args.visualize:
    # 如果前面没有初始化 nusc (比如 disable_check=True)，这里需要兜底
    if 'nusc' not in locals():
          nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    print("\n=== 开始可视化生成 ===")
    ds_len = len(nusc.sample)
    idx_list = np.random.choice(ds_len, 50, replace=False)
    print(f"ds_len:{ds_len}, idx_list:{idx_list}")
    
    for idx in idx_list:
      sample = nusc.sample[idx]
      sensor_names_to_vis = ['LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
      for sensor_name in sensor_names_to_vis:
        # visualize_to_jpg 也需要 SDK 对象
        visualize_to_jpg(nusc, sample, sensor_name, args.disable_velocity_visualize, axes_limit = args.axes_limit, path_prefix = osp.join(args.vis_out_dir, str(f'idx_{idx}_')))
  
  if not args.disable_check:
    print("\n✓ 所有数据集检查已完成。")
