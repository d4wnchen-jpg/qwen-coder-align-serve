# 实验结果与可追溯记录

> 本目录存 W2-W5 全部实测数据与图表。所有数字可复现：脚本 + 种子 + 版本号见下。

## 环境（可追溯）

| 项 | 值 |
|---|---|
| GPU | RTX 4090 24GB（AutoDL，内蒙古区） |
| vLLM | 0.27.1（V1 引擎） |
| torch | 2.13.0+cu130 |
| 基座模型 | Qwen/Qwen3-8B（bf16，本地 `/root/autodl-tmp/Qwen3-8B`） |
| 量化权重 | Qwen/Qwen3-8B-AWQ（官方，HF_HOME 缓存） |
| 评估框架 | EvalPlus（HumanEval+ 164 / MBPP+ 399，pass@1） |
| 生成参数 | `enable_thinking=false, temperature=0, max_tokens=2048, num_shots=1` |

## 数据文件

| 文件 | 内容 | 来源 |
|---|---|---|
| `serving_throughput.png` | 三组吞吐对比（bf16/AWQ/LoRA） | `scripts/make_figures.py` |
| `prefix_caching.png` | prefix caching 开关对照 | `scripts/make_figures.py` |
| `prompts_code.jsonl` | 代码风格压测提示词（服务器） | `serve/benchmark.sh` 生成 |

## 核心结果速查

### 对齐（SFT 矩阵，HumanEval+ / MBPP+）

| 模型 | loss | HumanEval+ | MBPP+ |
|---|---|---|---|
| 基线 | — | 81.1% | 61.7% |
| r16-e2 | 0.2299 | 73.2% | 62.9% |
| r16-e3 | 0.1506 | 71.3% | 60.2% |
| r64-e2 | 0.1683 | 75.6% | 61.2% |
| r64-e3 | 0.0644 | 70.7% | 60.7% |

### 量化（AWQ INT4 vs bf16）

| 指标 | bf16 | AWQ |
|---|---|---|
| 权重显存 | 16.72 GiB | 5.71 GiB |
| KV cache | 3.99 GiB | 14.0 GiB |
| HumanEval+ | 81.1% | 74.4% |
| MBPP+ | 61.2% | 60.9% |

### 部署压测

| 实验 | 配置 | 结果 |
|---|---|---|
| 吞吐基线 | random, RPS=8 | bf16 1290.84 / AWQ 1906.04 / LoRA 1123.98 tok/s |
| prefix caching | prefix_repetition, RPS=32 | 吞吐 1157→1853（+60%），TTFT 中位 10395→3887（-63%） |
| multi-LoRA | 4 adapter 轮询 vs 单 | 1068 vs 1124（-5%） |
| ngram 投机 | random, RPS=8 | 979.40（-24%，接受率 47%，已排除） |

## 复现命令

```bash
# 吞吐基线
bash serve/benchmark.sh 8000 base

# prefix caching 对照（起服务时显式开关）
#   关: vllm serve ... --no-enable-prefix-caching --port 8000
#   开: vllm serve ... --enable-prefix-caching  --port 8000
vllm bench serve --backend openai-chat --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions --model qwen3-8b \
  --tokenizer /root/autodl-tmp/Qwen3-8B \
  --dataset-name prefix_repetition --num-prompts 300 --request-rate 32 \
  --temperature 0 --save-result --result-dir results/<tag> --result-filename serve.json

# multi-LoRA 轮询
vllm bench serve ... --lora-assignment round-robin \
  --lora-modules r16-e2 r16-e3 r64-e2 r64-e3
```

> 注：`results/` 下的 `serve.json`（原始压测 JSON）在训练服务器上；本仓库保留图表 + 数据速查表，关键数字同步于 `docs/交接文档.md`。
