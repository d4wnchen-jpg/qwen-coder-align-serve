#!/usr/bin/env python3
"""用 vLLM OpenAI 兼容接口批量生成 HumanEval+ / MBPP+ 解答。

用法（先起服务，见 serve/serve_base.sh）:
    # 小样本验证（铁律：先跑 3 题看输出是否干净）
    python eval/gen_solutions.py --model qwen3-8b --limit 3 \
        --out samples/baseline/solutions.jsonl
    # 全量
    python eval/gen_solutions.py --model qwen3-8b \
        --out samples/baseline/solutions.jsonl

产出（按数据集拆成两个文件，避免 evalplus.evaluate 评分时 KeyError）:
    <out> 同名 + .humaneval.jsonl / .mbpp.jsonl

注意:
    Qwen3 是思考模型，默认会输出 <think> 推理块。评测必须关掉思考，
    否则答案被推理占满、代码被 max_tokens 截断、分数归零。
    已通过 chat_template_kwargs: {"enable_thinking": false} 关闭，
    另留 <think> 剥除作兜底。
"""
import argparse
import json
import re
from pathlib import Path

import requests

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def load_prompts() -> list:
    """从 evalplus 取 HumanEval+ 与 MBPP+ 的 prompt"""
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


def clean_code(code: str) -> str:
    """剥掉 <think> 块和 markdown 围栏，只留代码。"""
    code = strip_thinking(code)
    if "```" in code:
        parts = code.split("```")
        code = parts[1] if len(parts) > 1 else code
        code = re.sub(r"^[a-zA-Z]*\s*\n", "", code)  # 去语言标记行
    return code.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out", default="samples/humaneval_samples.jsonl")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--num-shots", type=int, default=1, help="pass@k 采样次数")
    ap.add_argument("--limit", type=int, default=None, help="只生成前 N 题（小样本验证）")
    args = ap.parse_args()

    prompts = load_prompts()
    if args.limit:
        prompts = prompts[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    he_out = out.with_name(out.stem + ".humaneval.jsonl")
    mbpp_out = out.with_name(out.stem + ".mbpp.jsonl")

    he_f = he_out.open("w", encoding="utf-8")
    mbpp_f = mbpp_out.open("w", encoding="utf-8")
    try:
        for p in prompts:
            for _ in range(args.num_shots):
                r = requests.post(
                    f"{args.base_url}/chat/completions",
                    json={
                        "model": args.model,
                        "messages": [{"role": "user", "content": p["prompt"]}],
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=600,
                )
                r.raise_for_status()
                code = r.json()["choices"][0]["message"]["content"]
                code = clean_code(code)
                f = he_f if p["task_id"].startswith("HumanEval/") else mbpp_f
                f.write(json.dumps({"task_id": p["task_id"], "solution": code},
                                   ensure_ascii=False) + "\n")
    finally:
        he_f.close()
        mbpp_f.close()

    print(f"生成完成 → {he_out} / {mbpp_out}（共 {len(prompts) * args.num_shots} 条）")
    print(f"评估: evalplus.evaluate --dataset humaneval --samples {he_out}")
    print(f"      evalplus.evaluate --dataset mbpp     --samples {mbpp_out}")


if __name__ == "__main__":
    main()
