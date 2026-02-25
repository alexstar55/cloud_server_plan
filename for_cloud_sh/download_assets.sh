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
# Cosmos 1.0 Prompt 500M (用于文本到世界生成)
hf download nvidia/Cosmos-1.0-Prompt-500M --local-dir ./Cosmos-1.0-Prompt-500M
# Cosmos 1.0 Diffusion 7B (用于视频生成)
hf download nvidia/Cosmos-1.0-Diffusion-7B-Video2World --local-dir ./Cosmos-1.0-Diffusion-7B-Video2World
# Diffusion Policy 依赖的 ResNet (通常在 torchvision 中，但如果使用预训练的 CLIP 等，也需要下载)
# huggingface-cli download openai/clip-vit-base-patch32 --local-dir ./clip-vit-base-patch32

# 3. 下载预训练权重 (ResNet101 for BEVFormer/mHC)
# echo "[INFO] 下载 ResNet101 预训练权重..."
# mkdir -p pretrained_weights
# wget -c https://download.pytorch.org/models/resnet101-63fe2227.pth -O pretrained_weights/resnet101-63fe2227.pth

# 4. 下载 LIBERO 数据集 (Phase 1 具身验证)
echo "[INFO] 下载 LIBERO 数据集 (10个任务的演示数据)..."
mkdir -p datasets/libero
# LIBERO 官方数据集托管在 HuggingFace 上
hf download YanjieZe/LIBERO --repo-type dataset --local-dir datasets/libero --include "libero_10/*"

echo "==== 资产下载完成 ===="
echo "请将 ${ASSETS_DIR} 整个文件夹打包上传至云服务器"