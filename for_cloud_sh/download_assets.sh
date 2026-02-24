#!/usr/bin/env bash
set -e

echo "==== 本地预下载预训练权重与数据集 ===="

ASSETS_DIR=${HOME}/workspace/offline_packages/assets
mkdir -p "${ASSETS_DIR}"
cd "${ASSETS_DIR}"

# 1. 安装 huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
  echo "[INFO] 安装 huggingface_hub..."
  pip install -U "huggingface_hub[cli]"
fi

# 2. 下载 HuggingFace 模型 (Cosmos / AlpaSim 可能需要的基座)
echo "[INFO] 下载 HuggingFace 模型..."
export HF_ENDPOINT=https://hf-mirror.com  # 使用国内镜像加速下载
# 示例：huggingface-cli download nvidia/Cosmos-1.0-Prompt-500M --local-dir ./Cosmos-1.0-Prompt-500M
echo "[WARN] 请在脚本中填入实际需要的 HF 模型 ID"

# 3. 下载预训练权重 (ResNet101 for BEVFormer/mHC)
echo "[INFO] 下载 ResNet101 预训练权重..."
mkdir -p pretrained_weights
wget -c https://download.pytorch.org/models/resnet101-63fe2227.pth -O pretrained_weights/resnet101-63fe2227.pth

# 4. 下载 LIBERO 数据集 (Phase 1 具身验证)
echo "[INFO] 下载 LIBERO 数据集 (10个任务的演示数据)..."
mkdir -p datasets/libero
# LIBERO 官方通常存放在 Google Drive 或 AWS，这里写一个占位符
# wget -c "<LIBERO_DATASET_URL>" -O datasets/libero/libero_10.hdf5
echo "[WARN] 请参考 LIBERO 官方 GitHub 获取实际的下载链接并替换此处的 wget 命令"

echo "==== 资产下载完成 ===="
echo "请将 ${ASSETS_DIR} 整个文件夹打包上传至云服务器"