#!/usr/bin/env bash
# 串行跑 4 组 SFT 实验矩阵。失败即停（set -e），带锁文件防重复启动。
set -euo pipefail

LOCK=/root/autodl-tmp/train.lock
if [ -f "$LOCK" ]; then
  echo "检测到训练锁 ${LOCK}，可能已有实例在跑。"
  echo "确认无残留后:  pkill -9 -f llamafactory; rm -f ${LOCK}"
  exit 1
fi
touch "$LOCK"
trap 'rm -f "$LOCK"' EXIT

for cfg in sft-r16-e2 sft-r16-e3 sft-r64-e2 sft; do
  echo "===== 开始训练 ${cfg} $(date '+%H:%M:%S') ====="
  llamafactory-cli train "train/${cfg}.yaml"
  echo "===== ${cfg} 完成 $(date '+%H:%M:%S') ====="
done
echo "===== 全部 4 组训练完成 ====="
