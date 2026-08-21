#!/usr/bin/env bash
# 压测入口：latency（离线吞吐）+ serve（在线 RPS 曲线）
# 用法: bash serve/benchmark.sh <port> <tag>
# 产出: results/<tag>/latency.json + serve.json（供 scripts/benchmark_report.py 生成报告）
set -euo pipefail

PORT="${1:-8000}"
TAG="${2:-base}"
BASE_URL="http://localhost:${PORT}"
OUT="results/${TAG}"
mkdir -p "${OUT}"

# 提示词集：优先用代码多轮风格（prefix caching 收益明显）
PROMPTS="${PROMPTS:-results/prompts_code.jsonl}"
if [ ! -f "${PROMPTS}" ]; then
  python - <<'PY'
import json
from pathlib import Path
Path("results").mkdir(exist_ok=True)
prompts = [
    "用 Python 实现一个线程安全的 LRU 缓存。",
    "补全以下函数：\ndef binary_search(arr, target):\n",
    "给下面这段代码写单元测试：\nclass Stack: ...",
    "解释这段代码的时间复杂度并优化：\nfor i in range(n):\n    for j in range(i, n):\n        ...",
    "把这段同步代码改成 asyncio 版本。",
] * 60
Path("results/prompts_code.jsonl").write_text(
    "\n".join(json.dumps({"prompt": p}, ensure_ascii=False) for p in prompts), encoding="utf-8")
print("生成 prompts_code.jsonl")
PY
fi

echo "==> 1) 在线服务吞吐（serve benchmark，异步并发，RPS 曲线）"
vllm bench serve \
  --backend vllm \
  --base-url "${BASE_URL}" \
  --model "$(curl -s ${BASE_URL}/v1/models | python -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')" \
  --dataset-name random \
  --input-len 512 \
  --output-len 256 \
  --num-prompts 300 \
  --request-rate 8 \
  --burstiness 2.0 \
  --seed 42 \
  --save-result \
  --result-dir "${OUT}" \
  --result-filename serve.json

echo "==> 2) 离线吞吐（throughput benchmark，需先停掉在线服务或另起引擎）"
vllm bench throughput \
  --model /root/autodl-tmp/Qwen3-8B \
  --dataset-name random \
  --input-len 512 \
  --output-len 256 \
  --num-prompts 300 \
  --save-result \
  --result-dir "${OUT}" \
  --result-filename throughput.json

echo "==> 压测完成 → ${OUT}/"
