# 数据说明

- **主数据源**: [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)（DeepSeek-Coder 系 OSS-Instruct 生成，代码指令，Apache-2.0 友好）
- **备选**: CodeAlpaca-20k、自行用 Qwen3 合成的领域指令（若做"私有代码库问答"，可加 RAG 检索数据，属于简历加分项）
- **清洗**: 见 `prepare_data.py`（去重、长度过滤、EvalPlus 关键词粗筛防污染）
- **合规**: 发布前检查数据源许可；训练产物 LoRA 权重可附自己的许可

## 下载与清洗（W1，本机 ¥0）

```bash
# 国内网络走镜像（海外直连去掉第一行即可）
export HF_ENDPOINT=https://hf-mirror.com
bash data/download.sh     # 下载去污染版 JSONL（~194MB）到 data/raw/

# 离线清洗 → data/train.json (3000) + data/dev.json (200) + data/report.json
python data/prepare_data.py --num-samples 3000 \
    --local-jsonl data/raw/magicoder-oss-instruct.jsonl
```

> 注：数据源本身已是 decontaminated 版本；本项目再叠加关键词粗筛防 EvalPlus 污染。

## 与 LLaMA-Factory 的对接

LLaMA-Factory 的 `dataset_info.json` 按文件路径自动识别 alpaca 格式（instruction/input/output），
本项目 `data/train.json` 即标准 alpaca 格式，直接把 `data/train.json` 填入 sft.yaml 的 dataset 即可。
