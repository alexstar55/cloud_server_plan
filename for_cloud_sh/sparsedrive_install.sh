#!/usr/bin/env bash
set -e

echo "==== 本地构建 SparseDrive Docker 镜像并导出 ===="

WORKDIR=${HOME}/workspace/offline_packages
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# 0. 增加 Git 缓冲区大小
git config --global http.postBuffer 524288000

# 1. 克隆 SparseDrive 仓库 (理想汽车 sdc-ai 官方库，使用国内代理)
if [ ! -d "SparseDrive" ] || [ ! -d "SparseDrive/.git" ]; then
  echo "[INFO] 克隆 SparseDrive 仓库..."
  rm -rf SparseDrive
  git clone https://github.moeyy.xyz/https://github.com/sdc-ai/SparseDrive.git
else
  echo "[INFO] SparseDrive 已存在，尝试更新"
  cd SparseDrive
  git remote set-url origin https://github.moeyy.xyz/https://github.com/sdc-ai/SparseDrive.git
  git pull || echo "[WARN] 更新失败，将使用现有代码继续"
  cd ..
fi

cd SparseDrive

# 2. 生成 Dockerfile
# SparseDrive 依赖 mmdet3d，基础镜像使用 PyTorch 2.4.0
cat <<EOF > Dockerfile.sparsedrive
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# 设置 pip 国内清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace/SparseDrive

# 替换 apt 源为阿里云（加速系统依赖安装）
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y git curl libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /workspace/SparseDrive/

# 安装基础 Python 依赖 (mmcv, mmdet3d 等通常在 requirements 中或需要后续手动编译)
RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt; else pip install numpy scipy tqdm einops opencv-python matplotlib; fi
EOF

# 3. 构建镜像
echo "[INFO] 开始构建 Docker 镜像 sparsedrive:v1 ..."
docker build -t sparsedrive:v1 -f Dockerfile.sparsedrive .

# 4. 导出镜像
echo "[INFO] 正在导出镜像到 ${WORKDIR}/sparsedrive_image.tar ..."
docker save -o ../sparsedrive_image.tar sparsedrive:v1

echo "==== SparseDrive 镜像构建与导出完成 ===="
echo "请将 ${WORKDIR}/sparsedrive_image.tar 上传至云服务器"