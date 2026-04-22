#!/bin/bash
# 监控两台 Mac 的 MILP 实验进展
# 每次运行追加一条状态到 runs/monitor.log
# 建议通过 cron 每 30 分钟跑一次

REPO=/Users/jjj/Desktop/工作/电力交易/energy-storage-rl
MAC2_HOST=jiang@192.168.0.21
MAC2_REPO=~/energy-storage-rl
LOG=$REPO/runs/monitor.log

cd $REPO

NOW=$(date '+%Y-%m-%d %H:%M:%S')
echo "" >> $LOG
echo "===== $NOW =====" >> $LOG

# --- Mac #1 状态 ---
echo "[Mac #1] (本机)" >> $LOG
MAC1_PID=$(pgrep -f "27_milp_shandong_all" | head -1)
if [ -n "$MAC1_PID" ]; then
  ELAPSED=$(ps -o etime= -p $MAC1_PID 2>/dev/null | tr -d ' ')
  CPU=$(ps -o %cpu= -p $MAC1_PID 2>/dev/null | tr -d ' ')
  echo "  ✅ Running (PID=$MAC1_PID, elapsed=$ELAPSED, CPU=${CPU}%)" >> $LOG
else
  echo "  ❌ NOT RUNNING" >> $LOG
fi

# 进度：数已完成的 pkl 文件
MAC1_DONE=$(ls $REPO/runs/milp_experiments/shandong_*.pkl 2>/dev/null | wc -l | tr -d ' ')
echo "  已完成：$MAC1_DONE / 13 实验" >> $LOG

# 最近 3 行日志
if [ -f $REPO/runs/milp_shandong.log ]; then
  echo "  最近日志：" >> $LOG
  tail -3 $REPO/runs/milp_shandong.log | sed 's/^/    /' >> $LOG
fi

# --- Mac #2 状态 ---
echo "" >> $LOG
echo "[Mac #2] (192.168.0.21)" >> $LOG
MAC2_STATUS=$(ssh -o ConnectTimeout=5 $MAC2_HOST '
  PID=$(pgrep -f "28_milp_other_provinces" | head -1)
  if [ -n "$PID" ]; then
    ELAPSED=$(ps -o etime= -p $PID 2>/dev/null | tr -d " ")
    CPU=$(ps -o %cpu= -p $PID 2>/dev/null | tr -d " ")
    echo "  ✅ Running (PID=$PID, elapsed=$ELAPSED, CPU=${CPU}%)"
  else
    echo "  ❌ NOT RUNNING"
  fi

  DONE=$(find ~/energy-storage-rl/runs/milp_experiments -name "*.pkl" 2>/dev/null | wc -l | tr -d " ")
  echo "  已完成：$DONE / 9 实验"

  if [ -f ~/energy-storage-rl/runs/milp_other.log ]; then
    echo "  最近日志："
    tail -3 ~/energy-storage-rl/runs/milp_other.log | sed "s/^/    /"
  fi
' 2>&1)
echo "$MAC2_STATUS" >> $LOG

# --- 汇总 ---
echo "" >> $LOG
TOTAL=$((MAC1_DONE + $(echo "$MAC2_STATUS" | grep "已完成" | grep -oE "[0-9]+" | head -1)))
echo "总进度：$TOTAL / 22 实验" >> $LOG

# 如果任一任务挂了，往一个 alert 文件里写
if [ -z "$MAC1_PID" ] || echo "$MAC2_STATUS" | grep -q "NOT RUNNING"; then
  echo "⚠️  ALERT at $NOW" >> $REPO/runs/ALERT.log
  echo "$MAC2_STATUS" >> $REPO/runs/ALERT.log
fi
