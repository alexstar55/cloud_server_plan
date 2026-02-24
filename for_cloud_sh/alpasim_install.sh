#!/usr/bin/env bash
set -e

echo "==== 本地拉取 alpaSim Docker 镜像并导出 ===="

WORKDIR=${HOME}/workspace/offline_packages
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# 1. 克隆仓库
if [ ! -d "alpasim" ]; then
  echo "[INFO] 克隆 alpaSim 仓库"
  git clone https://github.com/NVlabs/alpasim.git
else
  echo "[INFO] alpasim 目录已存在，尝试更新"
  cd alpasim && git pull && cd ..
fi

cd alpasim

# 2. 拉取/构建 docker-compose 中的镜像
echo "[INFO] 尝试拉取 alpaSim 依赖的 Docker 镜像..."
if command -v docker-compose &> /dev/null; then
  docker-compose pull || true
  docker-compose build || true
elif docker compose version &> /dev/null; then
  docker compose pull || true
  docker compose build || true
else
  echo "[ERROR] 本地未安装 docker-compose，无法解析镜像依赖"
  exit 1
fi

# 3. 导出镜像
echo "[INFO] 正在导出 alpasim 相关镜像到 ${WORKDIR}/alpasim_images.tar ..."
IMAGES=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -i "alpasim" || true)
if [ -z "$IMAGES" ]; then
  echo "[WARN] 未找到带有 alpasim 标签的镜像，请手动检查 docker-compose.yml 中的镜像名并导出。"
else
  docker save -o ../alpasim_images.tar $IMAGES
  echo "==== alpaSim 镜像导出完成 ===="
  echo "请将 ${WORKDIR}/alpasim_images.tar 上传至云服务器"
fi
