#!/bin/bash
# Mac #2 一键配置脚本
#
# 使用方法：
#   1. 在 Mac #1 上，把这个仓库 rsync 到 Mac #2:
#      rsync -avz --exclude '.venv' --exclude 'runs' --exclude '*.tar.gz' \
#        /Users/jjj/Desktop/工作/电力交易/energy-storage-rl/ \
#        jiang@192.168.0.21:~/energy-storage-rl/
#
#   2. SSH 到 Mac #2:
#      ssh jiang@192.168.0.21
#
#   3. 执行：
#      cd ~/energy-storage-rl
#      bash scripts/mac2_setup.sh

set -e

echo "=== Mac #2 环境配置 ==="

# Python 检查
echo "Python: $(python3 --version)"

# 安装依赖
echo "安装 Python 依赖..."
pip3 install --user \
  numpy pandas scipy scikit-learn \
  pyomo highspy \
  lightgbm \
  loguru \
  pyarrow

# 验证
python3 -c "
import numpy, pandas, scipy, sklearn
import pyomo.environ as pyo
import highspy
import lightgbm
import loguru
print('✅ All dependencies installed')

# 测试 HiGHS
m = pyo.ConcreteModel()
m.x = pyo.Var(bounds=(0,10))
m.obj = pyo.Objective(expr=m.x, sense=pyo.maximize)
m.c = pyo.Constraint(expr=m.x <= 5)
pyo.SolverFactory('appsi_highs').solve(m)
print(f'✅ HiGHS works: x={pyo.value(m.x)}')
"

# 检查 CPU 和内存
echo ""
echo "硬件："
sysctl -n machdep.cpu.brand_string
sysctl -n hw.ncpu hw.memsize | paste -sd' ' | awk '{printf "CPU: %s 核 / RAM: %.0f GB\n", $1, $2/1024/1024/1024}'

# 检查数据目录
if [ -d "data/china/processed" ]; then
    echo ""
    echo "数据目录："
    ls -lh data/china/processed/*.parquet 2>/dev/null | head -5 || echo "⚠️ 缺数据，需从 Mac #1 同步"
fi

echo ""
echo "=== 配置完成 ==="
echo ""
echo "下一步：运行跨省实验（3-5 小时）："
echo "  caffeinate -i python3 -u scripts/28_milp_other_provinces.py 2>&1 | tee runs/milp_other.log &"
echo ""
echo "防 Mac 睡眠："
echo "  sudo pmset -a sleep 0  # 永久关闭睡眠（跑完恢复 sudo pmset -a sleep 1）"
