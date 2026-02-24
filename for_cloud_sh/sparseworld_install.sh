#!/usr/bin/env bash
set -e

echo "==== 本地构建 SparseWorld Docker 镜像并导出 ===="

WORKDIR=${HOME}/workspace/offline_packages
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# 1. 克隆仓库
if [ ! -d "SparseWorld" ]; then
  git clone https://github.com/MSunDYY/SparseWorld.git
else
  echo "[INFO] SparseWorld 已存在，尝试更新"
  cd SparseWorld && git pull && cd ..
fi

cd SparseWorld

# 2. 生成 Dockerfile
cat <<EOF > Dockerfile.sparseworld
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# 设置国内源，方便后续在云端补充安装包
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace/SparseWorld

# 安装系统依赖
RUN apt-get update && apt-get install -y git curl libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /workspace/SparseWorld/

# 安装 Python 依赖
RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt; else pip install numpy scipy tqdm einops opencv-python; fi
RUN pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html || true
RUN pip install matplotlib plotly pyvista
EOF

# 3. 构建镜像
echo "[INFO] 开始构建 Docker 镜像 sparseworld:v1 ..."
docker build -t sparseworld:v1 -f Dockerfile.sparseworld .

# 4. 导出镜像
echo "[INFO] 正在导出镜像到 ${WORKDIR}/sparseworld_image.tar ..."
docker save -o ../sparseworld_image.tar sparseworld:v1

echo "==== SparseWorld 镜像构建与导出完成 ===="
echo "请将 ${WORKDIR}/sparseworld_image.tar 上传至云服务器"
