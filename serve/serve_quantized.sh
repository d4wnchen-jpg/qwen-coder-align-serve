#!/usr/bin/env bash
# 量化服务：AWQ INT4（4090 24GB 可跑）或 FP8
# AWQ 权重：官方 Qwen/Qwen3-8B-AWQ（已存在）；如需自行量化用 autoawq / llm-compressor
set -euo pipefail

MODE="${1:-awq}"   # awq | fp8

if [ "$MODE" = "awq" ]; then
  MODEL="Qwen/Qwen3-8B-AWQ"
  QARGS="--quantization awq_marlin"
elif [ "$MODE" = "fp8" ]; then
  MODEL="Qwen/Qwen3-8B-Instruct"
  QARGS="--quantization fp8"
else
  echo "用法: $0 [awq|fp8]"; exit 1
fi

VLLM_USE_V1=1 vllm serve "${MODEL}" \
  --served-model-name qwen3-8b-quant \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --port 8001 \
  ${QARGS}
