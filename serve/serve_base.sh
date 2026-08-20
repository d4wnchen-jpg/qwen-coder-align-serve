#!/usr/bin/env bash
# 基线服务：Qwen3-8B-Instruct（BF16，4090 24GB 直跑）
# 验证版本: vLLM 0.19+（以租卡机实际安装版本为准）
set -euo pipefail

VLLM_USE_V1=1 vllm serve Qwen/Qwen3-8B-Instruct \
  --served-model-name qwen3-8b \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000
# 注意(W5 压测): 基线服务刻意不开 prefix caching，作为干净对照组；
# "+prefix caching" 组另起服务时加回 --enable-prefix-caching 再压测对比。
# 备选（显存紧张时）:
#   --quantization awq_marlin   （需 AWQ 权重，见 serve_quantized.sh）
#   --tensor-parallel-size 2    （双卡）
