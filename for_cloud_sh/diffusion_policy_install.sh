#!/usr/bin/env bash
set -e

echo "==== 本地构建 Diffusion Policy Docker 镜像并导出 ===="

WORKDIR=${HOME}/workspace/offline_packages
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# 1. 克隆仓库
if [ ! -d "diffusion_policy" ]; then
  git clone https://github.moeyy.xyz/https://github.com/real-stanford/diffusion_policy.git
else
  echo "[INFO] diffusion_policy 已存在，尝试更新"
  cd diffusion_policy && git pull && cd ..
fi

cd diffusion_policy

# 2. 生成 Dockerfile (DP 依赖较多，建议用 conda 基础镜像或直接 pip)
cat <<EOF > Dockerfile.dp
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
WORKDIR /workspace/diffusion_policy

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y git curl libgl1-mesa-glx libglib2.0-0 build-essential && rm -rf /var/lib/apt/lists/*

COPY . /workspace/diffusion_policy/

# 安装依赖 (根据官方 conda.yaml 提取的核心 pip 包)
RUN pip install zarr wandb hydra-core dill imageio
RUN pip install diffusers==0.11.1 av numba gym
EOF

# 3. 构建与导出
echo "[INFO] 开始构建 Docker 镜像 diffusion_policy:v1 ..."
docker build -t diffusion_policy:v1 -f Dockerfile.dp .

# echo "[INFO] 正在导出镜像到 ${WORKDIR}/diffusion_policy_image.tar ..."
# docker save -o ../diffusion_policy_image.tar diffusion_policy:v1

# echo "==== Diffusion Policy 镜像构建与导出完成 ===="