#!/usr/bin/env bash
# Multi-LoRA 服务：一个基座挂 N 个领域 LoRA，运行时热切换（简历亮点）
# 前置: train/run_sft.sh 已导出 adapter 到 outputs/lora-export/
set -euo pipefail

# LoRA 模块列表: name=path（可加多个领域，如 code-review / unit-test / doc-gen）
LORA_MODULES=(
  "code-align=$(pwd)/outputs/lora-export/sft-r64-e3"
  # "code-review=$(pwd)/outputs/lora-export/review-lora"
)

MODULES_ARG=""
for m in "${LORA_MODULES[@]}"; do
  MODULES_ARG="${MODULES_ARG} --lora-modules ${m}"
done

VLLM_USE_V1=1 vllm serve Qwen/Qwen3-8B-Instruct \
  --served-model-name qwen3-8b-multilora \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --gpu-memory-utilization 0.90 \
  --port 8002 \
  ${MODULES_ARG}

# 调用示例（OpenAI 接口，指定 LoRA 名）:
#   curl http://localhost:8002/v1/chat/completions -d '{
#     "model": "code-align",
#     "messages": [{"role":"user","content":"写一个快排"}]
#   }'
