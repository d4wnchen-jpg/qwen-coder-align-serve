# -*- coding: utf-8 -*-
"""压测报告生成：results/<tag>/*.json → Markdown 表格 + 对比图

用法:
    python scripts/benchmark_report.py --results results/
产出:
    results/report.md           # 汇总表格（base / cache / mtp / multilora 横向对比）
    results/report.png          # 吞吐 / TTFT / TPOT 对比图
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文（无中文字体环境自动回退英文标签）
try:
    from matplotlib import font_manager
    for f in ["/System/Library/Fonts/Hiragino Sans GB.ttc"]:
        try:
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            CN = True
            break
        except Exception:
            CN = False
except Exception:
    CN = False

L = lambda zh, en: zh if CN else en  # noqa: E731


def load_latency(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {
        "output_tokens_per_second": d.get("output_tokens_per_second"),
        "input_tokens_per_second": d.get("input_tokens_per_second"),
        "total_tokens_per_second": d.get("total_tokens_per_second"),
    }


def load_serve(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {
        "ttft_p50": d.get("ttft_p50"),
        "ttft_p95": d.get("ttft_p95"),
        "tpot_p50": d.get("tpot_p50"),
        "request_throughput": d.get("request_throughput"),
        "output_throughput": d.get("output_throughput"),
        "mean_request_rate": d.get("mean_request_rate"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = Path(args.results)

    tags, latency, serve = [], {}, {}
    for tag_dir in sorted(root.iterdir()):
        if not tag_dir.is_dir():
            continue
        lf, sf = tag_dir / "latency.json", tag_dir / "serve.json"
        if lf.exists():
            tags.append(tag_dir.name)
            latency[tag_dir.name] = load_latency(lf)
        if sf.exists():
            serve[tag_dir.name] = load_serve(sf)

    if not tags:
        print("未找到 results/*/latency.json — 先跑 serve/benchmark.sh")
        return

    # ---- Markdown ----
    lines = ["# 压测报告", "", f"标签: {', '.join(tags)}", ""]
    lines += ["## 离线吞吐", "", "| 配置 | 输出 tokens/s | 总 tokens/s |", "|---|---|---|"]
    for t in tags:
        if t in latency:
            x = latency[t]
            lines.append(f"| {t} | {x['output_tokens_per_second']:.0f} | {x['total_tokens_per_second']:.0f} |")
    lines += ["", "## 在线服务（RPS 负载）", "", "| 配置 | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p50 (ms) | 请求吞吐 (req/s) |", "|---|---|---|---|---|"]
    for t in tags:
        if t in serve:
            x = serve[t]
            lines.append(f"| {t} | {x['ttft_p50']:.1f} | {x['ttft_p95']:.1f} | {x['tpot_p50']:.1f} | {x['request_throughput']:.2f} |")
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"→ {root / 'report.md'}")

    # ---- 图表 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")

    # 吞吐对比（相对提升）
    ax = axes[0]
    vals = [latency[t]["output_tokens_per_second"] for t in tags if t in latency]
    bars = ax.bar(tags, vals, color="#58a6ff")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom", color="white")
    ax.set_title(L("输出吞吐 (tokens/s)", "Output throughput (tok/s)"), color="white")
    ax.tick_params(colors="white")

    ax = axes[1]
    vals = [serve[t]["ttft_p50"] for t in tags if t in serve]
    ax.bar(tags, vals, color="#f0883e")
    ax.set_title(L("TTFT p50 (ms)", "TTFT p50 (ms)"), color="white")
    ax.tick_params(colors="white")

    ax = axes[2]
    vals = [serve[t]["tpot_p50"] for t in tags if t in serve]
    ax.bar(tags, vals, color="#3fb950")
    ax.set_title(L("TPOT p50 (ms)", "TPOT p50 (ms)"), color="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(root / "report.png", dpi=120, facecolor=fig.get_facecolor())
    print(f"→ {root / 'report.png'}")


if __name__ == "__main__":
    main()
