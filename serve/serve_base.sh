#!/usr/bin/env bash
# 基线服务：Qwen3-Coder-30B-A3B-Instruct（BF16，4090 需开启 CPU offload 或换量化版，见 serve_quantized.sh）
# 验证版本: vLLM >= 0.8.x（MTP 支持需较新版本）
set -euo pipefail

VLLM_USE_V1=1 vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --served-model-name qwen3-coder \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --port 8000
# 备选（显存紧张时）:
#   --quantization awq_marlin   （需 AWQ 权重，见 serve_quantized.sh）
#   --tensor-parallel-size 2    （双卡）
