# Qwen-Coder Align & Serve

**代码领域 LLM 对齐与高吞吐推理系统** —— 一条完整流水线：数据清洗 → QLoRA SFT → (DPO) → 量化 → vLLM 高吞吐部署（前缀缓存 / MTP 投机解码 / Multi-LoRA）→ 基准评估与压测报告。

> English: A full-pipeline project for domain-adapted code LLM alignment and high-throughput serving: data curation → QLoRA SFT → DPO → quantization → vLLM serving (prefix caching / MTP speculative decoding / multi-LoRA) → benchmark & load-test reports.

## 技术栈

| 环节 | 选型 | 说明 |
|---|---|---|
| 基座模型 | **Qwen/Qwen3-8B-Instruct**（4090 bf16 直跑） | 通用 8B → 代码域 SFT 提升 |
| 对齐 | **LLaMA-Factory**（QLoRA SFT + DPO） | 社区事实标准 |
| 数据 | Magicoder-OSS-Instruct（筛 2-4k 高质量子集） | 代码指令数据 |
| 评估 | **EvalPlus**（HumanEval+/MBPP+，pass@1） | 防数据污染标准 |
| 量化 | AWQ INT4 / FP8 | |
| 服务 | **vLLM**：continuous batching + prefix caching + EAGLE3/ngram 投机解码 + multi-LoRA | 高吞吐核心（Qwen3-8B 无原生 MTP） |

## 目录结构

```
qwen-coder-align-serve/
├── README.md                  # 本文件
├── docs/执行计划.md            # 6 周执行表 + 租卡预算（¥300 档）
├── data/
│   ├── prepare_data.py        # 数据下载/清洗/切分 → LLaMA-Factory 格式
│   └── README.md              # 数据来源与合规说明
├── train/
│   ├── sft.yaml               # QLoRA SFT 配置
│   ├── dpo.yaml               # DPO 配置（可选，W4）
│   └── run_sft.sh             # 一键训练入口
├── eval/
│   ├── gen_solutions.py       # 用 vLLM OpenAI 接口批量生成 HumanEval+ 答案
│   └── run_evalplus.sh        # EvalPlus 评估入口
├── serve/
│   ├── serve_base.sh          # 基线服务
│   ├── serve_quantized.sh     # AWQ/FP8 量化服务
│   ├── serve_lora.sh          # Multi-LoRA 多模型热切换服务
│   └── benchmark.sh           # vLLM bench 压测（latency + serve RPS）
├── scripts/
│   └── benchmark_report.py    # 压测结果 → Markdown 报告 + 图表
├── requirements.txt
└── .gitignore
```

## 快速开始（在租用 GPU 机器上）

```bash
# 1. 环境
pip install -r requirements.txt

# 2. 数据（本机/任意机器均可）
python data/prepare_data.py --num-samples 3000

# 3. 训练（4090 24GB 可跑，约 6-10 小时）
bash train/run_sft.sh

# 4. 部署压测
bash serve/serve_base.sh          # 终端 A：起服务
bash serve/benchmark.sh           # 终端 B：压测（输出结果到 results/）

# 5. 评估
bash eval/run_evalplus.sh

# 6. 报告
python scripts/benchmark_report.py --results results/
```

## 6 周路线（详见 docs/执行计划.md）

- **W1** 数据与脚本准备（本机，¥0）
- **W2** 基座基线（vLLM + EvalPlus + 吞吐基线）
- **W3** QLoRA SFT 实验矩阵
- **W4** DPO（可选）+ AWQ 量化 + 复测
- **W5** 高吞吐实验：prefix caching / MTP / multi-LoRA 逐项对比
- **W6** 压测报告 + GitHub 包装

## 简历条目模板

> **代码领域 LLM 对齐与高吞吐推理系统**（Qwen3-8B / vLLM / LLaMA-Factory）
> 主导数据清洗 → QLoRA SFT → AWQ 量化 → Multi-LoRA 热切换服务全流程；
> EvalPlus 基准提升 X%；开启前缀缓存与 MTP 投机解码后吞吐提升 X 倍；
> 单卡同时服务 N 个领域 LoRA。

## License

MIT（数据与模型遵循各自原始许可）。
