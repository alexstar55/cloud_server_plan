#!/usr/bin/env bash
set -e

echo "==== 本地构建 DriveVGGT / DVGT Docker 镜像并导出 ===="

WORKDIR=${HOME}/workspace/offline_packages
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# 1. 克隆仓库
if [ ! -d "DVGT" ]; then
  git clone https://github.com/wzzheng/DVGT.git
else
  echo "[INFO] DVGT 已存在，尝试更新"
  cd DVGT && git pull && cd ..
fi

cd DVGT

# 2. 生成 Dockerfile
cat <<EOF > Dockerfile.drivevggt
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace/DVGT

RUN apt-get update && apt-get install -y git curl libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY . /workspace/DVGT/

RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt; else pip install numpy pillow tqdm opencv-python matplotlib einops; fi
RUN pip install kornia sympy trimesh pyvista
EOF

# 3. 构建镜像
echo "[INFO] 开始构建 Docker 镜像 drivevggt:v1 ..."
docker build -t drivevggt:v1 -f Dockerfile.drivevggt .

# 4. 导出镜像
echo "[INFO] 正在导出镜像到 ${WORKDIR}/drivevggt_image.tar ..."
docker save -o ../drivevggt_image.tar drivevggt:v1

echo "==== DriveVGGT 镜像构建与导出完成 ===="
echo "请将 ${WORKDIR}/drivevggt_image.tar 上传至云服务器"
