#!/usr/bin/env bash
# 下载 Magicoder-OSS-Instruct-75K（官方去污染版 JSONL，约 194MB）
# 用法:
#   国内网络:  HF_ENDPOINT=https://hf-mirror.com bash data/download.sh
#   海外直连:  bash data/download.sh
set -euo pipefail
cd "$(dirname "$0")"

BASE="${HF_ENDPOINT:-https://huggingface.co}"
REMOTE="datasets/ise-uiuc/Magicoder-OSS-Instruct-75K/resolve/main/data-oss_instruct-decontaminated.jsonl"
OUT="raw/magicoder-oss-instruct.jsonl"

mkdir -p raw
if [ -f "$OUT" ]; then
    echo "已存在: $OUT（跳过下载）"
else
    echo "下载: $BASE/$REMOTE"
    curl -L --fail --retry 3 -o "$OUT" "$BASE/$REMOTE"
fi
echo "done: $OUT ($(du -h "$OUT" | cut -f1))"
