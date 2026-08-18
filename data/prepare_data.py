# -*- coding: utf-8 -*-
"""数据准备：Magicoder-OSS-Instruct → 清洗 → LLaMA-Factory 格式

用法（本机/任意机器，无需 GPU）:
    # 方式 A：HF Hub 流式（国内网络先 export HF_ENDPOINT=https://hf-mirror.com）
    python data/prepare_data.py --num-samples 3000
    # 方式 B：离线（先 bash data/download.sh 下好 JSONL）
    python data/prepare_data.py --num-samples 3000 --local-jsonl data/raw/magicoder-oss-instruct.jsonl

产出:
    data/train.json        # [{"instruction":..., "input":..., "output":...}]
    data/dev.json          # 200 条验证集
    data/report.json       # 清洗统计
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent

# 清洗规则（代码数据专用）
MAX_INST_LEN = 2000   # 指令过长（噪声）丢弃
MAX_OUT_LEN = 4000
MIN_OUT_LEN = 50
# 与 EvalPlus 测试题近重复的样本（防污染）——关键词粗筛
POLLUTION_HINTS = [
    "def has_close_elements", "def separate_paren_groups", "def truncate_number",
    "def below_zero", "def rolling_max", "def is_palindrome",
]


def normalize(s: str) -> str:
    s = re.sub(r"\r\n", "\n", s or "")
    return s.strip()


def is_quality(sample: dict) -> bool:
    inst = normalize(sample.get("instruction", ""))
    inp = normalize(sample.get("input", ""))
    out = normalize(sample.get("output", ""))
    if not inst or not out:
        return False
    if len(inst) > MAX_INST_LEN or len(out) > MAX_OUT_LEN or len(out) < MIN_OUT_LEN:
        return False
    # 丢弃与 EvalPlus 测试题近重复的样本（粗筛，正式版可加 embedding 去重）
    blob = (inst + out).lower()
    if any(h in blob for h in POLLUTION_HINTS):
        return False
    return True


def dedup_key(sample: dict) -> str:
    return hashlib.md5(
        (normalize(sample.get("instruction", "")) + normalize(sample.get("output", ""))).encode()
    ).hexdigest()


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=3000)
    ap.add_argument("--dev-size", type=int, default=200)
    ap.add_argument("--dataset", default="ise-uiuc/Magicoder-OSS-Instruct-75K")
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--local-jsonl",
        default=None,
        help="本地 JSONL 文件路径（离线清洗用；优先于 --dataset）",
    )
    return ap.parse_args()


def iter_raw(args):
    """按行产出原始样本 dict（HF 流式 or 本地 JSONL）。"""
    if args.local_jsonl:
        print(f"读取本地 JSONL: {args.local_jsonl}")
        with open(args.local_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    from datasets import load_dataset

    print(f"下载 {args.dataset} ({args.split}) ...")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    yield from ds


def to_instruction_triple(ex: dict) -> dict:
    """Magicoder 格式(problem/solution) 与 LLaMA-Factory 格式(instruction/output) 双兼容。"""
    return {
        "instruction": normalize(ex.get("instruction") or ex.get("problem") or ""),
        "input": normalize(ex.get("input", "") or ""),
        "output": normalize(ex.get("output") or ex.get("response") or ex.get("solution") or ""),
    }


def main():
    args = _parse_args()

    seen, kept = set(), []
    for raw in iter_raw(args):
        ex = to_instruction_triple(raw)
        if not is_quality(ex):
            continue
        k = dedup_key(ex)
        if k in seen:
            continue
        seen.add(k)
        kept.append(ex)
        if len(kept) >= args.num_samples + args.dev_size:
            break

    dev = kept[: args.dev_size]
    train = kept[args.dev_size :]

    (DATA_DIR / "train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (DATA_DIR / "dev.json").write_text(
        json.dumps(dev, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "kept": len(kept),
        "train": len(train),
        "dev": len(dev),
        "note": "已去重+长度过滤+EvalPlus关键词粗筛；正式版建议增加语义去重",
    }
    (DATA_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"完成: train={len(train)} dev={len(dev)} → {DATA_DIR}")


if __name__ == "__main__":
    main()
