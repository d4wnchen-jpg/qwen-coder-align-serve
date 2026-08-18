#!/usr/bin/env python3
"""用 vLLM OpenAI 兼容接口批量生成 HumanEval+ 解答

用法（先起服务，见 serve/serve_base.sh 或 serve/serve_lora.sh）:
    python eval/gen_solutions.py \
        --base-url http://localhost:8000/v1 \
        --model qwen3-coder \
        --out samples/sft-r64-e3/humaneval_samples.jsonl

依赖: pip install evalplus
"""
import argparse
import json
from pathlib import Path

import requests


def load_prompts() -> list:
    """从 evalplus 取 HumanEval+ 与 MBPP+ 的 prompt"""
    import evalplus.data  # noqa
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    prompts = []
    for problem_id, prob in get_human_eval_plus().items():
        prompts.append({
            "task_id": problem_id,
            "prompt": prob["prompt"],
            "entry_point": prob["entry_point"],
        })
    for problem_id, prob in get_mbpp_plus().items():
        prompts.append({
            "task_id": problem_id,
            "prompt": prob["prompt"],
        })
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3-coder")
    ap.add_argument("--out", default="samples/humaneval_samples.jsonl")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--num-shots", type=int, default=1, help="pass@k 采样次数")
    args = ap.parse_args()

    prompts = load_prompts()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for p in prompts:
            for _ in range(args.num_shots):
                r = requests.post(
                    f"{args.base_url}/chat/completions",
                    json={
                        "model": args.model,
                        "messages": [{"role": "user", "content": p["prompt"]}],
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                    },
                    timeout=600,
                )
                r.raise_for_status()
                code = r.json()["choices"][0]["message"]["content"]
                # 只保留代码块（如果模型包了 markdown）
                if "```" in code:
                    code = code.split("```")[1]
                    if code.startswith("python"):
                        code = code[len("python"):]
                f.write(json.dumps({"task_id": p["task_id"], "solution": code}, ensure_ascii=False) + "\n")

    print(f"生成完成 → {out}（共 {len(prompts) * args.num_shots} 条）")
    print(f"评估: evalplus.evaluate --dataset humaneval --samples {out}")
    print(f"      evalplus.evaluate --dataset mbpp --samples {out}")


if __name__ == "__main__":
    main()
