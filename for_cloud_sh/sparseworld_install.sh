#!/usr/bin/env bash
set -e

echo "==== SparseWorld 环境搭建开始 ===="

WORKDIR=${HOME}/workspace
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

# 2. 创建 conda 环境
if command -v conda &> /dev/null; then
  ENV_NAME=sparseworld
  conda create -y -n ${ENV_NAME} python=3.11
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ${ENV_NAME}
else
  echo "[ERROR] 未检测到 conda，请先安装再运行此脚本"
  exit 1
fi

# 3. 安装 PyTorch（同样按实际 CUDA 版本调整）
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 4. 项目依赖
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  # 若没有，则按常见 occupancy/world model 项目手动补
  pip install numpy scipy tqdm einops opencv-python
fi

# 5. 可加上一些稀疏/几何相关库（视项目实际需要保留/删减）
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html || true
pip install matplotlib plotly pyvista

# 6. 写 env.sh
cat <<EOF > env.sh
#!/usr/bin/env bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}
export PYTHONPATH=$(pwd):\$PYTHONPATH
EOF
chmod +x env.sh

echo "==== SparseWorld 环境搭建完成 ===="
echo "后续使用："
echo "  cd ${WORKDIR}/SparseWorld"
echo "  source env.sh"
echo "  python scripts/demo_xxx.py   # 根据仓库实际脚本名称替换"
