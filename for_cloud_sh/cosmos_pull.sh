#!/usr/bin/env bash
set -e

echo "==== 本地构建 Cosmos Transfer Docker 镜像并导出 ===="

WORKDIR=${HOME}/workspace/offline_packages
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# 0. 增加 Git 缓冲区大小
git config --global http.postBuffer 524288000

# 1. 克隆仓库 (使用国内代理)
if [ ! -d "cosmos-transfer1" ] || [ ! -d "cosmos-transfer1/.git" ]; then
  echo "[INFO] 克隆 cosmos-transfer1 仓库..."
  rm -rf cosmos-transfer1
  git clone https://github.moeyy.xyz/https://github.com/nvidia-cosmos/cosmos-transfer1.git
else
  echo "[INFO] cosmos-transfer1 已存在，尝试更新"
  cd cosmos-transfer1
  git remote set-url origin https://github.moeyy.xyz/https://github.com/nvidia-cosmos/cosmos-transfer1.git
  git pull || echo "[WARN] 更新失败，将使用现有代码继续"
  cd ..
fi

cd cosmos-transfer1
# 更新子模块
git submodule update --init --recursive || echo "[WARN] 子模块更新失败，请检查网络"

# 2. 修改 Dockerfile 以适应国内网络
echo "[INFO] 修改 Dockerfile 注入国内源..."
# 备份原版
cp Dockerfile Dockerfile.bak

# 替换基础镜像为华为云缓存 (如果可用)，并注入 apt 和 pip 国内源
sed -i 's|FROM nvcr.io/nvidia/tritonserver:25.04-vllm-python-py3|FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/nvcr.io/nvidia/tritonserver:25.04-vllm-python-py3|g' Dockerfile
# 如果华为云没有缓存这个特定的 tritonserver 镜像，可能还是会失败。
# 作为备选，如果构建失败，请手动将上一行注释掉，恢复使用 nvcr.io 官方源，并确保本地网络能访问 nvcr.io。

# 在 FROM 之后插入国内源配置
sed -i '/FROM/a \
# 设置 pip 国内清华源\n\
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple\n\
# 替换 apt 源为阿里云\n\
RUN sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list \&\& \\\n\
    sed -i "s/security.ubuntu.com/mirrors.aliyun.com/g" /etc/apt/sources.list' Dockerfile

# 3. 构建镜像
echo "[INFO] 开始构建 Docker 镜像 cosmos-transfer1:v1 ..."
docker build -t cosmos-transfer1:v1 -f Dockerfile .

# 4. 导出镜像
echo "[INFO] 正在导出镜像到 ${WORKDIR}/cosmos_transfer_image.tar ..."
docker save -o ../cosmos_transfer_image.tar cosmos-transfer1:v1

echo "==== Cosmos 镜像构建与导出完成 ===="
echo "请将 ${WORKDIR}/cosmos_transfer_image.tar 上传至云服务器"