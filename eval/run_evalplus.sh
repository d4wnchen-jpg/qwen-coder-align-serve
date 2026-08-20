#!/usr/bin/env bash
# EvalPlus 评估入口：先起 vLLM 服务，再生成 + 评测
set -euo pipefail

MODEL="${1:-qwen3-8b}"                    # 服务名（需与 serve_base.sh 的 --served-model-name 一致）
TAG="${2:-sft-r64-e3}"                    # 结果目录标签
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"

echo "==> 生成 HumanEval+ / MBPP+ 解答"
python eval/gen_solutions.py --base-url "${BASE_URL}" --model "${MODEL}" \
    --out "samples/${TAG}/humaneval_samples.jsonl"

echo "==> HumanEval+ 评估"
evalplus.evaluate --dataset humaneval --samples "samples/${TAG}/humaneval_samples.jsonl"

echo "==> MBPP+ 评估"
evalplus.evaluate --dataset mbpp --samples "samples/${TAG}/humaneval_samples.jsonl"

echo "==> 完成，结果在 evalplus 缓存目录（默认 ~/.evalplus）"
