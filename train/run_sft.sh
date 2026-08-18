#!/usr/bin/env bash
# 一键 SFT（在租用的 GPU 机器上执行）
set -euo pipefail

CFG="${1:-train/sft.yaml}"

echo "==> LLaMA-Factory SFT: ${CFG}"
llamafactory-cli train "${CFG}"

echo "==> 合并 LoRA 到 base（导出用于 vLLM multi-LoRA 服务的 adapter）"
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --adapter_name_or_path outputs/sft-r64-e3 \
  --template qwen3 \
  --finetuning_type lora \
  --export_dir outputs/lora-export/sft-r64-e3 \
  --export_size 5 \
  --export_legacy_format false

echo "==> 完成。adapter 位于 outputs/lora-export/sft-r64-e3"
