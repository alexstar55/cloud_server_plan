# 夜间自动跑批 Runbook（一个月版）

适用镜像：
- alpasim:v1
- sparsedrive:v1
- diffusion_policy:v1

目标：
- 夜间无人值守连续运行
- 失败可自动重试
- 次日可快速复盘（日志、资源、产出）

约束：
- 仅命令框架与检查点，不包含业务代码

---

## 0. 目录与日志约定（先执行）

```bash
mkdir -p ~/nightly/{configs,logs,pids,reports,artifacts}
mkdir -p ~/nightly/logs/{alpasim,sparsedrive,diffusion_policy}
mkdir -p ~/nightly/artifacts/{alpasim,sparsedrive,diffusion_policy}
```

检查点：
- logs 与 artifacts 目录创建成功
- 目录权限正确（当前用户可写）

---

## 1. 启动前检查（每晚必做）

```bash
# 1) GPU 与驱动
nvidia-smi

# 2) Docker 可用性
docker ps

# 3) 镜像存在
docker images | grep -E "alpasim|sparsedrive|diffusion_policy"

# 4) 磁盘与内存
df -h
free -h
```

检查点：
- 两张 GPU 可见
- 三个镜像均存在
- 磁盘剩余空间 >= 300G（建议值）

---

## 2. 容器启动模板（固定框架）

### 2.1 alpasim（GPU 0）

```bash
docker run -d --rm \
  --name alpasim_nightly \
  --gpus '"device=0"' \
  --shm-size 32g \
  -v ~/workspace/alpasim:/workspace/alpasim \
  -v ~/nightly/artifacts/alpasim:/workspace/output \
  -v ~/nightly/logs/alpasim:/workspace/logs \
  alpasim:v1 \
  bash -lc "cd /workspace/alpasim && <ALPASIM_RUN_CMD>"
```

### 2.2 sparsedrive（GPU 1）

```bash
docker run -d --rm \
  --name sparsedrive_nightly \
  --gpus '"device=1"' \
  --shm-size 32g \
  -v ~/workspace/SparseWorld-TC:/workspace/SparseWorld-TC \
  -v ~/nightly/artifacts/sparsedrive:/workspace/output \
  -v ~/nightly/logs/sparsedrive:/workspace/logs \
  sparsedrive:v1 \
  bash -lc "cd /workspace/SparseWorld-TC && <SPARSEDRIVE_RUN_CMD>"
```

### 2.3 diffusion_policy（GPU 0 或 1，按当晚安排）

```bash
docker run -d --rm \
  --name dp_nightly \
  --gpus '"device=0"' \
  --shm-size 24g \
  -v ~/workspace/diffusion_policy:/workspace/diffusion_policy \
  -v ~/nightly/artifacts/diffusion_policy:/workspace/output \
  -v ~/nightly/logs/diffusion_policy:/workspace/logs \
  diffusion_policy:v1 \
  bash -lc "cd /workspace/diffusion_policy && <DP_RUN_CMD>"
```

检查点：
- docker ps 中 1~2 个任务容器正常运行
- GPU 资源占用符合预期（避免两个大任务抢同一张卡）

---

## 3. 日志采集与心跳监控（夜间自动）

### 3.1 实时查看

```bash
docker logs -f --tail 200 alpasim_nightly
docker logs -f --tail 200 sparsedrive_nightly
docker logs -f --tail 200 dp_nightly
```

### 3.2 每 10 分钟写入心跳（可放 cron）

```bash
*/10 * * * * date >> ~/nightly/reports/heartbeat.log
*/10 * * * * nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader >> ~/nightly/reports/gpu_heartbeat.log
*/10 * * * * docker ps --format "table {{.Names}}\t{{.Status}}" >> ~/nightly/reports/container_heartbeat.log
```

检查点：
- heartbeat.log 每 10 分钟有新增
- 任务容器状态无频繁重启

---

## 4. 失败自动重试模板（最多 1 次）

```bash
# 统一模板：失败后自动重试一次
bash -lc '<RUN_CMD>' || (sleep 30 && bash -lc '<RUN_CMD>')
```

容器内框架可写成：

```bash
bash -lc "cd <WORKDIR> && (<RUN_CMD> || (sleep 30 && <RUN_CMD>))"
```

检查点：
- 失败日志中可区分 first-fail 与 retry-run
- 重试后仍失败则任务终止并保留错误日志

---

## 5. 次日早晨复盘清单（10分钟）

```bash
# 1) 容器最终状态
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}"

# 2) GPU 夜间利用率尾部
cat ~/nightly/reports/gpu_heartbeat.log | tail -n 30

# 3) 关键日志尾部
tail -n 100 ~/nightly/logs/alpasim/*.log 2>/dev/null
tail -n 100 ~/nightly/logs/sparsedrive/*.log 2>/dev/null
tail -n 100 ~/nightly/logs/diffusion_policy/*.log 2>/dev/null

# 4) 产出文件计数
find ~/nightly/artifacts -type f | wc -l
```

检查点：
- 是否有有效产出（文件数、时间戳、大小）
- 是否出现 NaN/OOM/断连等关键字
- 是否需要当天白天优先修复

---

## 6. 当天白天修复优先级（固定顺序）

1) 先修中断类问题（容器启动失败、权限、挂载路径）
2) 再修资源类问题（OOM、显存冲突、I/O 瓶颈）
3) 最后修指标类问题（参数、收敛、切片波动）

---

## 7. 一键夜跑脚本框架（建议）

```bash
# 仅框架示例：按当晚任务启停
bash nightly_precheck.sh
bash nightly_launch_alpasim.sh
bash nightly_launch_sparsedrive.sh
# 或替换为 diffusion_policy
bash nightly_watchdog.sh
```

建议最小脚本集合：
- nightly_precheck.sh（环境检查）
- nightly_launch_*.sh（按镜像启动任务）
- nightly_watchdog.sh（心跳+异常记录）
- morning_summary.sh（早晨汇总）

---

## 8. 风险提示

- 如果 docker.sock 权限问题复发，优先修复用户组权限，避免每次 sudo。
- 如果 GitHub 拉取不稳定，尽量复用已拉取目录，不在夜间临时 clone 大仓库。
- cosmos 链路未稳定前，不把其作为夜跑硬依赖，避免拖慢主线周更产出。
