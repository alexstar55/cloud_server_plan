#!/usr/bin/env bash
set -e

echo "==== DriveVGGT / DVGT 环境搭建开始 ===="

WORKDIR=${HOME}/workspace
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# 1. 克隆仓库（以 DVGT 为例，DriveVGGT 多半是类似结构）
if [ ! -d "DVGT" ]; then
  git clone https://github.com/wzzheng/DVGT.git
else
  echo "[INFO] DVGT 已存在，尝试更新"
  cd DVGT && git pull && cd ..
fi

cd DVGT

# 2. 创建 conda 环境
if command -v conda &> /dev/null; then
  ENV_NAME=drivevggt
  conda create -y -n ${ENV_NAME} python=3.11
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ${ENV_NAME}
else
  echo "[ERROR] 未检测到 conda，请先在服务器上安装 Anaconda / Miniconda 再运行此脚本"
  exit 1
fi

# 3. 安装 PyTorch（示例使用 CUDA 12.1，你可按服务器实际情况改）
echo "[INFO] 安装 PyTorch（请根据服务器 CUDA 版本调整）"
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 4. 安装项目依赖
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  # 若仓库无统一 requirements，可手动装基本包
  pip install numpy pillow tqdm opencv-python matplotlib einops
fi

# 5. 一些几何/可视化辅助库（发挥你数学直觉用）
pip install kornia sympy trimesh pyvista

# 6. 设置 PYTHONPATH（可写入到一个 env.sh 中，方便以后 source）
cat <<EOF > env.sh
#!/usr/bin/env bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}
export PYTHONPATH=$(pwd):\$PYTHONPATH
EOF
chmod +x env.sh

echo "==== DriveVGGT / DVGT 环境搭建完成 ===="
echo "后续使用："
echo "  cd ${WORKDIR}/DVGT"
echo "  source env.sh"
echo "  python your_demo_or_train_script.py"
