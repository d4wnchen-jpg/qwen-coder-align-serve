#!/usr/bin/env bash
# 量化服务：AWQ INT4（4090 24GB 可跑）或 FP8
# AWQ 权重：优先用社区现成（HF 搜 "Qwen3-Coder-30B-A3B-AWQ"），
#           没有则自行量化: autoawq / llm-compressor（记录在 docs/执行计划.md 风险表）
set -euo pipefail

MODE="${1:-awq}"   # awq | fp8

if [ "$MODE" = "awq" ]; then
  MODEL="casperhansen/Qwen3-Coder-30B-A3B-Instruct-AWQ"   # TODO: 换成实际存在的社区权重或自量化路径
  QARGS="--quantization awq_marlin"
elif [ "$MODE" = "fp8" ]; then
  MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
  QARGS="--quantization fp8"
else
  echo "用法: $0 [awq|fp8]"; exit 1
fi

VLLM_USE_V1=1 vllm serve "${MODEL}" \
  --served-model-name qwen3-coder-quant \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --port 8001 \
  ${QARGS}
