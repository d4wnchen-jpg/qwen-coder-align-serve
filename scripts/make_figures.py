# -*- coding: utf-8 -*-
"""生成项目核心图表（数据来自 W5 实测，见 docs/交接文档.md）

产出:
    results/serving_throughput.png    # 吞吐对比（bf16 / AWQ / LoRA）
    results/prefix_caching.png        # prefix caching 开关对照（吞吐 + TTFT）
用法:
    python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体（无则回退英文标签）
try:
    from matplotlib import font_manager
    _f = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    font_manager.fontManager.addfont(_f)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
    CN = True
except Exception:
    CN = False

L = lambda zh, en: zh if CN else en  # noqa: E731

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

DARK = "#0d1117"
PANEL = "#161b22"
BLUE = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#f0883e"
RED = "#f85149"


def _style(ax):
    ax.set_facecolor(PANEL)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def _annotate(ax, bars, vals, fmt="{:.0f}", color="white", dy=0):
    for b, v in zip(bars, vals):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(v),
                ha="center", va="bottom", color=color, fontsize=9)


# ---- 图 1: 吞吐对比 ----
fig, ax = plt.subplots(figsize=(7, 4.2))
fig.patch.set_facecolor(DARK)
_labels = ["bf16", "AWQ INT4", "LoRA\n(r64-e2)"]
_vals = [1290.84, 1906.04, 1123.98]
_colors = [BLUE, GREEN, ORANGE]
bars = ax.bar(_labels, _vals, color=_colors)
_annotate(ax, bars, _vals, dy=20)
ax.set_title(L("输出吞吐对比（random 数据集, RPS=8）",
               "Output throughput (random, RPS=8)"), pad=12)
ax.set_ylabel(L("tokens/s", "tokens/s"))
ax.set_ylim(0, 2200)
_style(ax)
fig.tight_layout()
fig.savefig(RESULTS / "serving_throughput.png", dpi=130, facecolor=DARK)
plt.close(fig)

# ---- 图 2: prefix caching 开关对照（高负载 RPS=32）----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
fig.patch.set_facecolor(DARK)

# 吞吐
bars = ax1.bar([L("关缓存", "off"), L("开缓存", "on")], [1157.54, 1853.12],
               color=[ORANGE, GREEN])
_annotate(ax1, bars, [1157.54, 1853.12], dy=30)
ax1.set_title(L("吞吐 (tokens/s)", "Throughput (tok/s)"), pad=10)
ax1.set_ylim(0, 2200)
ax1.set_ylabel(L("tokens/s", "tokens/s"))
_style(ax1)

# TTFT 中位
bars = ax2.bar([L("关缓存", "off"), L("开缓存", "on")], [10395, 3887],
               color=[ORANGE, GREEN])
_annotate(ax2, bars, [10395, 3887], fmt="{:.0f}", dy=200)
ax2.set_title(L("TTFT 中位 (ms)", "Median TTFT (ms)"), pad=10)
ax2.set_ylim(0, 12000)
ax2.set_ylabel(L("ms", "ms"))
_style(ax2)

fig.suptitle(L("Prefix Caching 对照（prefix_repetition 数据集, RPS=32 高负载）",
               "Prefix caching (prefix_repetition, RPS=32)"),
             color="white", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS / "prefix_caching.png", dpi=130, facecolor=DARK)
plt.close(fig)

print(f"→ {RESULTS / 'serving_throughput.png'}")
print(f"→ {RESULTS / 'prefix_caching.png'}")
