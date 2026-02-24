# 面向云服务器与个人执行：Runbook（两个月）

> 目的：把“数据生成 → 统一导出 → 训练/推理 → 切片评测 → 失败挖掘 → 再生成”做成可重复的闭环。
>
> 约束：云服务器可联网但无法访问国外网络；下载走本地 VPN 机器离线转运。

---

## 0) 成功标准（两个月必须交付）

- 数据：累计生成并可用 **≥1000 组**仿真样本（6cam+1lidar，时间/位姿对齐，NuScenes 格式可读，含元数据标签）。
- 切片：至少覆盖并可控过采样
  - 距离：重点 50→80m（远距）
  - 遮挡：有标签（必须纳入）
  - 类别：`bus` 为重点（漏检更严重）
- 指标：对既有评测体系输出
  - 全量 mAP/NDS（守门不退化）
  - 距离分桶/遮挡切片的 recall 或 mAP（目标：远距稳定提升）
- 工程：`generate → export → validate → report` 一键化，可交接。

---

## 1) 服务器开通当天（Day 1）检查清单

### 1.1 硬件与驱动

- `nvidia-smi`：确认双 5880、驱动版本、显存大小
- `nvidia-smi topo -m`：确认 PCIe/NVLink 关系（决定数据并行/推理绑卡策略）

### 1.2 Docker + GPU 容器能力（为 cosmos 做前置）

- `docker --version`
- `nvidia-container-toolkit` 已安装且生效：
  - 运行一个 CUDA base 镜像，验证容器内 `nvidia-smi` 正常

### 1.3 磁盘/IO

- 预留目录：
  - `data/raw_sim/`（仿真原始输出）
  - `data/nuscenes_sim/`（导出后的 NuScenes）
  - `data/manifests/`（每个数据版本的 manifest）
  - `reports/`（评测报告与图表）

---

## 2) 离线转运（无国外网）默认流程

### 2.1 本地（有 VPN）准备包

- 公开数据/权重（若需要）：下载到本地后打包（例如 `tar.zst`）
- Docker 镜像：
  - `docker pull <image>`
  - `docker save -o <name>.tar <image>`

### 2.2 上传到云服务器

- 使用公司允许的上传方式（scp/对象存储/网盘/内网制品库）

### 2.3 云端导入

- Docker：`docker load -i <name>.tar`
- 数据解包到 `data/` 下，并记录版本号到 `data/manifests/`

---

## 3) 必须包含：cosmos-transfer1-carla Docker 部署与验收

### 3.1 启动命令（按组长示例）

- `docker run -d --shm-size 96g --gpus=all --ipc=host -p 8080:8080 cosmos-transfer1-carla`

说明：
- `--shm-size 96g` 用于大共享内存；实际大小可按容器需求调整
- 端口可能不止 8080：以镜像说明为准

### 3.2 验收步骤（按顺序）

1) 容器存活：`docker ps` 可见
2) GPU 可见：容器内 `nvidia-smi` 正常
3) 服务可达：`curl http://127.0.0.1:8080`（或按镜像的 health endpoint）
4) 生成/导出最小产物：至少 1 个 Sequence 的传感器输出（哪怕还没 NuScenes 化）
5) 若镜像目标为 transfer-to-CARLA：验证 CARLA 内可回放

产物落盘要求：
- 所有 demo 生成结果必须写入 `data/raw_sim/cosmos/<date>_<run_id>/`
- 必须有 `meta.json` 记录：镜像 tag、启动参数、commit/hash（若有）、时间范围

---

## 4) 主线闭环：数据生成 → 导出 → 校验 → 评测

### 4.1 场景规格（Scenario Spec）

- 文件格式：YAML/JSON，版本化
- 必备字段：
  - 参与者类别与数量（含 `bus`）
  - 初始相对距离（覆盖 50–80m，重点 60–80m）
  - 相对速度（匹配 robobus 20–42 km/h）
  - 遮挡配置（使用遮挡属性标签或生成时打标签）
  - 传感器配置（6cam+1lidar，内外参与时间同步模型与实车一致）

### 4.2 导出器（Exporter）

目标：把任意仿真输出转成“你现有的 NuScenes 转换脚本可直接吃”的输入形态。

- 最低要求：时间戳单调、ego pose 连续、相机/LiDAR 同步关系可回溯
- 强制接入：你们的定位连续性检查规则（GAP/TOLERANCE），避免隐性错配

### 4.3 版本管理（强制）

每次生成的数据版本都要有：
- `manifest.json`：
  - 数据版本号
  - scenario_spec 版本/commit
  - 生成器类型（CARLA/Isaac/alpaSim/cosmos）
  - 覆盖统计（距离/类别/遮挡）
  - 导出器版本

---

## 5) 评测与切片（对齐你们现有 mAP/NDS，同时补证据链）

### 5.1 必做切片

- 距离 bins：0–20 / 20–40 / 40–50 / 50–65 / 65–80（可按你们实现便利调整）
- 遮挡：按已有遮挡属性标签
- 类别重点：`bus` 单独报表（recall、FN rate）

### 5.2 守门阈值（建议写死，避免“远距变好全量变差”）

- 全量 NDS 不低于基线 -0.2（示例阈值，实际按你们现网容忍度改）
- 远距关键切片指标必须稳定提升（用 bootstrap CI 做稳定性判断）

---

## 6) 两个月周节奏（可执行）

- Week 0（开通前）：冻结协议与目录结构；准备离线包（数据/权重/镜像）
- Week 1：最小链路跑通（生成 10–20 seq → 导出 NuScenes → 切片报告）
- Week 2：6cam+1lidar 对齐与校验器做稳
- Week 3：场景库 V1（累计 300–500），重点远距+遮挡+bus
- Week 4：第一次混合训练/推理验证（你输出报告与归因）
- Week 5–6：闭环两轮（失败挖掘驱动再生成，累计 ≥1000）
- Week 7：一键化与交接（脚本/manifest/报告模板）
- Week 8：总结与下一期资源申请依据（ROI、风险、下一步）

---

## 7) 止损规则（两个月不被工程细节吞掉）

- 任一生成器平台若连续 2 天业余时间仍无法产出可回放的最小 Sequence：降级为只跑你们最熟的平台（优先 CARLA），保证闭环不断。
- 任一导出版本若出现时间戳/pose 大规模错配：立即回滚到上一个可用 manifest，先保证评测链路与报告持续产出。
- 不把“照片级真实感”设为验收目标；只以“切片覆盖与指标闭环”为中心。
