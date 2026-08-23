#!/usr/bin/env bash
# 量化服务：AWQ INT4（4090 24GB 可跑）或 FP8
# AWQ 权重：官方 Qwen/Qwen3-8B-AWQ（已下载到 HF_HOME 缓存）
# 注意: 起服务前需 export HF_HOME=/root/autodl-tmp/hf_cache（权重在缓存里，走 repo id 加载）
#       如需自行量化用 autoawq / llm-compressor
set -euo pipefail

MODE="${1:-awq}"   # awq | fp8

if [ "$MODE" = "awq" ]; then
  MODEL="Qwen/Qwen3-8B-AWQ"
  QARGS="--quantization awq_marlin"
elif [ "$MODE" = "fp8" ]; then
  MODEL="Qwen/Qwen3-8B"
  QARGS="--quantization fp8"
else
  echo "用法: $0 [awq|fp8]"; exit 1
fi

# 注: 不开 prefix caching，作为与基线(serve_base.sh bf16)同条件的干净对照；
#     量化评测就是要公平对比「同样的服务配置下，量化掉多少分」。
# 注: vLLM 0.27 V1 是唯一引擎，不再需要 VLLM_USE_V1=1。
vllm serve "${MODEL}" \
  --served-model-name qwen3-8b-quant \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8001 \
  ${QARGS}
