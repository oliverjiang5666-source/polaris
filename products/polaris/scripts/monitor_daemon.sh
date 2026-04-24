#!/bin/bash
# 后台监控守护进程。每 30 分钟跑一次 monitor_experiments.sh
# 用法：
#   nohup bash scripts/monitor_daemon.sh > /dev/null 2>&1 &
#   disown

REPO=/Users/jjj/Desktop/工作/电力交易/energy-storage-rl

while true; do
  /bin/bash $REPO/scripts/monitor_experiments.sh
  sleep 1800  # 30 分钟
done
