#!/usr/bin/env bash
set -e

echo "==== 本地拉取/构建 alpaSim Docker 镜像并导出 ===="

WORKDIR=${HOME}/workspace/offline_packages
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# 0. 增加 Git 缓冲区大小，防止大文件拉取超时
git config --global http.postBuffer 524288000

# 1. 克隆仓库 (加入失败清理和国内镜像加速)
# 检查目录是否存在且是否是一个完整的 git 仓库
if [ ! -d "alpasim" ] || [ ! -d "alpasim/.git" ]; then
  echo "[INFO] 清理可能损坏的目录并重新克隆 alpaSim 仓库..."
  rm -rf alpasim
  # 使用 kkgithub 镜像加速克隆
  git clone https://kkgithub.com/NVlabs/alpasim.git
else
  echo "[INFO] alpasim 目录已存在，尝试更新"
  cd alpasim
  # 切换到镜像源并更新
  git remote set-url origin https://kkgithub.com/NVlabs/alpasim.git
  git pull || echo "[WARN] 更新失败，将使用现有代码继续"
  cd ..
fi

cd alpasim

# 2. 拉取/构建镜像
echo "[INFO] 尝试构建 alpaSim 依赖的 Docker 镜像..."

# 检查是否存在 docker-compose 文件
if [ -f "docker-compose.yml" ] || [ -f "docker-compose.yaml" ]; then
  echo "[INFO] 发现 docker-compose 文件，使用 compose 构建..."
  if command -v docker-compose &> /dev/null; then
    docker-compose pull || true
    docker-compose build || true
  else
    docker compose pull || true
    docker compose build || true
  fi
else
  echo "[INFO] 未找到 docker-compose 文件，自动生成默认 Dockerfile..."
  cat <<EOF > Dockerfile.alpasim
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace/alpasim

RUN apt-get update && apt-get install -y git curl libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY . /workspace/alpasim/

RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi
EOF
  docker build -t alpasim:v1 -f Dockerfile.alpasim .
fi

# 3. 导出镜像
echo "[INFO] 正在导出 alpasim 相关镜像到 ${WORKDIR}/alpasim_images.tar ..."
IMAGES=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -i "alpasim" || true)

if [ -z "$IMAGES" ]; then
  echo "[WARN] 未找到带有 alpasim 标签的镜像，导出失败。"
else
  docker save -o ../alpasim_images.tar $IMAGES
  echo "==== alpaSim 镜像导出完成 ===="
  echo "请将 ${WORKDIR}/alpasim_images.tar 上传至云服务器"
fi