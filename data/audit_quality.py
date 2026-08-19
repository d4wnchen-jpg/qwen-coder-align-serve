# -*- coding: utf-8 -*-
"""数据质量审计：对 data/train.json 与 data/dev.json 做 7 类常见问题扫描。

用法:
    python data/audit_quality.py [train.json dev.json ...]

检查项:
  1. unbalanced_fence : 输出中 ``` 数量为奇数（代码块未闭合，可能截断）
  2. no_code_block    : 输出没有任何代码块（纯文字解答/拒答）
  3. skeleton         : 解答是占位骨架（pass/TODO/NotImplementedError 占比过高）
  4. truncated        : 尾部明显截断（以 "..." 结尾）
  5. refusal          : 输出是拒答/要更多信息的开头
  6. lang_mismatch    : 题目点名某语言但代码块是另一种语言
  7. dup_instruction  : 相同 instruction 出现多次（跨样本统计）
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare_data import extract_code_blocks  # noqa: E402  # 与清洗脚本共用同一解析


def audit_file(path: Path, inst_counter: collections.Counter):
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = collections.Counter()
    flags = []
    for idx, s in enumerate(data):
        inst, out = s["instruction"], s["output"]
        issues = []

        n_fence = out.count("```")
        if n_fence % 2 == 1:
            issues.append("unbalanced_fence")
        blocks = extract_code_blocks(out)
        if not blocks:
            issues.append("no_code_block")
        else:
            lang, code = extract_code_blocks(out)[0]
            lines = [l.strip() for l in code.split("\n")]
            code_lines = [l for l in lines if l and not l.startswith("#")]
            ph = [
                l
                for l in lines
                if re.fullmatch(
                    r"(pass|\.\.\.|raise NotImplementedError|#\s*TODO.*|"
                    r"#\s*Your code here|#\s*Add your code here|"
                    r"#\s*Implement this.*)",
                    l,
                    re.I,
                )
            ]
            if (len(ph) >= 2 and code_lines and len(ph) / len(code_lines) >= 0.3) or (
                ph and len(code_lines) <= 2
            ):
                issues.append("skeleton")
            langm = re.search(
                r"\b(python|java|c\+\+|javascript|rust|golang|\bgo\b|c#|swift|php|typescript)\b",
                inst,
                re.I,
            )
            if (
                langm
                and lang
                and langm.group(1).lower() not in (lang, "golang" if lang == "go" else "")
            ):
                issues.append("lang_mismatch")

        if out.rstrip().endswith("..."):
            issues.append("truncated")
        if re.match(
            r"^\s*(I('m| am) sorry|As an AI|I cannot|I can't|"
            r"I need more information|Unfortunately)",
            out,
            re.I,
        ):
            issues.append("refusal")

        inst_counter[inst.strip()] += 1
        if issues:
            flags.append((idx, issues, inst[:90]))

    for _, issues, _ in flags:
        for it in issues:
            stats[it] += 1
    return data, stats, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=["data/train.json", "data/dev.json"])
    args = ap.parse_args()

    inst_counter: collections.Counter = collections.Counter()
    for f in args.files:
        data, stats, flags = audit_file(Path(f), inst_counter)
        print(f"\n{'=' * 66}\n{f}  (n={len(data)})")
        for k in [
            "unbalanced_fence", "no_code_block", "skeleton", "truncated",
            "refusal", "lang_mismatch",
        ]:
            print(f"  {k:18s} {stats[k]:5d}  ({stats[k]/len(data)*100:.1f}%)")
        print("  示例（每类最多 2 条）:")
        shown = set()
        for idx, issues, head in flags:
            key = tuple(issues)
            if key in shown or len(shown) >= 12:
                continue
            shown.add(key)
            print(f"    #{idx} {issues} | {head}")

    print(f"\n{'=' * 66}")
    dup = {k: v for k, v in inst_counter.items() if v > 1}
    print(f"重复 instruction 数: {len(dup)}（出现 {sum(dup.values())} 次）")
    if dup:
        k, v = max(dup.items(), key=lambda kv: kv[1])
        print(f"  最多的一条出现 {v} 次: {k[:80]}")


if __name__ == "__main__":
    main()
