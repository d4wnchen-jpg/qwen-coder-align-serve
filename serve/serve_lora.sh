#!/usr/bin/env bash
# Multi-LoRA 服务：一个基座挂 N 个 LoRA，运行时热切换（简历亮点）
# 前置: train/run_sft.sh 已完成训练，adapter 在 outputs/<group>/（训练直接产物，无需 export 合并）
# 注意: 服务器上需 sed 模型名 Qwen/Qwen3-8B -> /root/autodl-tmp/Qwen3-8B（同 serve_base.sh 惯例）
set -euo pipefail

# LoRA 模块列表: name=path（评测期挂 4 组超参 adapter；最终演示可换不同领域 LoRA）
LORA_MODULES=(
  "r16-e2=$(pwd)/outputs/sft-r16-e2"
  "r16-e3=$(pwd)/outputs/sft-r16-e3"
  "r64-e2=$(pwd)/outputs/sft-r64-e2"
  "r64-e3=$(pwd)/outputs/sft-r64-e3"
)

# 关键: --lora-modules 在 vLLM 0.27 的 nargs='+'，必须一个 flag 接多个值（空格分隔），
#       不能重复写多个 --lora-modules（否则 argparse 只保留最后一个，/v1/models 只剩 1 个 adapter）
MODULES_ARG="--lora-modules ${LORA_MODULES[*]}"

# 注: 不开 prefix caching，作为与基线(serve_base.sh)同条件的干净对照；
#     W5 压测 prefix caching 收益时另起服务加回 --enable-prefix-caching。
# 注: vLLM 0.27 V1 是唯一引擎，不再需要 VLLM_USE_V1=1。
vllm serve Qwen/Qwen3-8B \
  --served-model-name qwen3-8b-multilora \
  --max-model-len 8192 \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --gpu-memory-utilization 0.90 \
  --port 8002 \
  ${MODULES_ARG}

# 调用示例（OpenAI 接口，指定 LoRA 名）:
#   curl http://localhost:8002/v1/chat/completions -d '{
#     "model": "r64-e3",
#     "messages": [{"role":"user","content":"写一个快排"}]
#   }'
