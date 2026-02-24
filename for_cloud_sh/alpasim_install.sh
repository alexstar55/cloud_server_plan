#!/usr/bin/env bash
set -e

echo "==== alpaSim 环境搭建开始 ===="

# 0. 基础依赖：git、curl
sudo apt-get update
sudo apt-get install -y git curl

# 1. 检查 Docker
if ! command -v docker &> /dev/null; then
  echo "[INFO] 未检测到 Docker，开始安装（如公司有统一镜像/源，按公司规范修改）"
  curl -fsSL https://get.docker.com | bash
  sudo usermod -aG docker "$USER"
  echo "[INFO] Docker 已安装，建议重新登录终端以生效 docker 组权限"
fi

# 2. 检查 docker-compose
if ! command -v docker-compose &> /dev/null; then
  echo "[INFO] 安装 docker-compose"
  sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-$(uname -s)-$(uname -m)" \
       -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
fi

# 3. 克隆仓库
WORKDIR=${HOME}/workspace
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

if [ ! -d "alpasim" ]; then
  echo "[INFO] 克隆 alpaSim 仓库"
  git clone https://github.com/NVlabs/alpasim.git
else
  echo "[INFO] alpasim 目录已存在，尝试更新"
  cd alpasim
  git pull
  cd ..
fi

cd alpasim

# 4.（可选）创建 Python 虚拟环境，用于工具脚本
if command -v conda &> /dev/null; then
  echo "[INFO] 使用 conda 创建 alpasim 工具环境"
  conda create -y -n alpasim-tools python=3.11
  # 注意：非交互脚本中激活 conda 一般用 source
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate alpasim-tools
  pip install -r requirements.txt || true
else
  echo "[WARN] 未检测到 conda，略过本地 Python 环境创建，你可以之后手动处理"
fi

# 5. 生成 .env 模板，方便你填 HuggingFace Token 等
if [ ! -f ".env" ]; then
  cat <<EOF > .env
# === alpaSim 环境变量示例 ===
# 必须：你的 Hugging Face token，用于下载模型/资产
HF_TOKEN=your_hf_token_here

# 其他可选配置视官方文档填写
EOF
  echo "[INFO] 已生成 .env 模板，请手动填入 HF_TOKEN 等信息"
fi

echo "==== alpaSim 环境脚本已完成基础配置 ===="
echo "后续在服务器上，你可以执行："
echo "  cd ${WORKDIR}/alpasim"
echo "  docker-compose up -d   # 启动仿真服务"
