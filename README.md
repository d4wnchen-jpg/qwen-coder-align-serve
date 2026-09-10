# Qwen-Coder Align & Serve

**代码领域 LLM 对齐与高吞吐推理系统** —— 一条完整流水线：数据清洗 → LoRA SFT 实验矩阵 → AWQ 量化 → vLLM 高吞吐部署（prefix caching / multi-LoRA）→ EvalPlus 评估与压测报告。

> English: An end-to-end project for code-domain LLM alignment and high-throughput serving: data curation → LoRA SFT matrix → AWQ quantization → vLLM serving (prefix caching / multi-LoRA) → EvalPlus evaluation & load-test reports.

## 系统架构

```
                    ┌─────────────────────────────────────────────┐
                    │              数据层（W1，本机）              │
                    │  Magicoder-OSS-Instruct-75K → 清洗/去重/防污染 │
                    │              train 3000 / dev 200            │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │            训练层（W3，4090）                │
                    │   LLaMA-Factory · LoRA(bf16+sdpa)            │
                    │   矩阵 r16/64 × e2/3 → 4 个 adapter           │
                    │   结论：强基座 SFT 负优化（过拟合）           │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │          模型资产（W4）                      │
                    │  ├─ Qwen3-8B bf16（基线 16.7G）              │
                    │  ├─ AWQ INT4（5.7G，省 66%）                 │
                    │  └─ r64-e2 LoRA adapter（最优）              │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │        服务层（W5，vLLM 0.27.1 V1）          │
                    │  ├─ prefix caching：吞吐 +60% / TTFT -63%    │
                    │  ├─ multi-LoRA：一基座挂 4 adapter 热切换    │
                    │  └─ 投机解码：ngram 负优化（-24%，排除）     │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────────┐
                    │        评估层（贯穿全程）                    │
                    │  EvalPlus：HumanEval+ 164 / MBPP+ 399       │
                    │  vLLM bench：吞吐 / TTFT / TPOT 压测         │
                    └─────────────────────────────────────────────┘
```

## 核心结果（一图流）

### 对齐：SFT 在强基座上的负优化

| 模型 | 训练 loss | HumanEval+ | MBPP+ |
|---|---|---|---|
| 基线 Qwen3-8B | — | **81.1%** | 61.7% |
| r16-e2 | 0.2299 | 73.2% | 62.9% |
| r16-e3 | 0.1506 | 71.3% | 60.2% |
| **r64-e2** | 0.1683 | **75.6%** | 61.2% |
| r64-e3 | 0.0644 | 70.7% | 60.7% |


### 量化：AWQ INT4 的 tradeoff

| 指标 | bf16 | AWQ INT4 |
|---|---|---|
| 权重显存 | 16.72 GiB | **5.71 GiB（省 66%）** |
| KV cache 可用 | 3.99 GiB | **14.0 GiB（多 3.5×）** |
| HumanEval+ | 81.1% | 74.4%（↓6.7） |
| MBPP+ | 61.2% | 60.9%（↓0.3） |

### 部署压测：三项优化 + 一项负结果

| 优化项 | 结果 | 采用 |
|---|---|---|
| AWQ 量化 | 输出吞吐 +48%（1291→1906 tok/s） | ✅ |
| prefix caching | 吞吐 +60% / TTFT 中位 -63%（高负载 RPS=32） | ✅ |
| multi-LoRA | 挂 4 个 adapter ≈ 挂 1 个（-5%，≈噪声） | ✅ |
| ngram 投机解码 | **-24%（负优化，接受率仅 47%）** | ❌ 有数据排除 |

## 技术栈

| 环节 | 选型 | 说明 |
|---|---|---|
| 基座模型 | **Qwen/Qwen3-8B**（bf16，4090 直跑） | 8B，代码能力接近 SOTA |
| 对齐 | **LLaMA-Factory 0.9.5**（LoRA，bf16 + sdpa） | 训练矩阵 r16/64 × e2/3 |
| 数据 | Magicoder-OSS-Instruct-75K（筛 3000 train / 200 dev） | 防污染清洗 |
| 评估 | **EvalPlus**（HumanEval+ 164 / MBPP+ 399，pass@1） | 防背题标准 |
| 量化 | AWQ INT4（官方权重） | 省显存 66% |
| 服务 | **vLLM 0.27.1**（V1 引擎） | prefix caching + multi-LoRA |

## 目录结构

```
qwen-coder-align-serve/
├── README.md                  # 本文件（核心结果 + 复现）
├── data/                      # prepare_data.py + dataset_info.json
├── train/                     # sft.yaml / sft-r16-e2.yaml ... / run_matrix.sh
├── eval/                      # gen_solutions.py / run_evalplus.sh
├── serve/                     # serve_base.sh / serve_quantized.sh / serve_lora.sh / benchmark.sh
├── scripts/benchmark_report.py
└── requirements.txt
```

## 快速复现

```bash
# 0. 环境（torch 2.13 + CUDA 13 + vLLM 0.27.1）
pip install -r requirements.txt

# 1. 数据
python data/prepare_data.py --num-samples 3000

# 2. 训练（4090 24GB，bf16 LoRA + sdpa，约 36min/组）
bash train/run_matrix.sh

# 3. 部署 + 压测
bash serve/serve_base.sh          # 起基线服务
bash serve/benchmark.sh 8000 base # 压测 → results/

# 4. 评估
bash eval/run_evalplus.sh

# 5. 报告
python scripts/benchmark_report.py --results results/
```


## License

MIT（数据与模型遵循各自原始许可）。
