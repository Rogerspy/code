# 代码生成统一规范 (FlagOS / 含光 PPU)

> 所有 5 个模型的训练/推理代码遵循本规范。生成代码前务必先读 `checkpoints/<model>/README.md` 与 `config.json`，按文档 faithfully 实现。

## 1. 运行环境
- 硬件：含光 PPU (PPU-ZW810E)。在 Aliyun `inference-xpu-pytorch` 容器内 `torch.cuda.is_available()==True`（PPU 伪装成 CUDA），**一切按 CUDA 处理**。
- 容器内：torch 2.x + cu，transformers（已支持 Qwen3），vllm 0.18。`sentence-transformers`、`FlagEmbedding` 会在容器内 pip 安装。
- 权重路径：宿主 `/home/qiyiyan/songchao/checkpoints/<model>`；容器内挂载到 `/workspace/checkpoints/<model>`。
- 代码必须从环境变量读 checkpoints 根目录，默认值用宿主路径，使其在宿主/容器都可用：
  ```python
  import os
  CKPT_ROOT = os.environ.get("CHECKPOINTS_DIR", "/home/qiyiyan/songchao/checkpoints")
  MODEL_PATH = os.path.join(CKPT_ROOT, "<model>")
  ```

## 2. 设备处理
```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
```
优先使用 sentence-transformers / FlagEmbedding（自动选 cuda）。手动张量显式 `.to(device)`。

## 3. 目录结构（每个模型，位于 `./code/<model>/`）
- `test_cases.json` — 20 条测试用例（格式见 §4）
- `train.py` — 在 20 条用例上微调，保存到 `./finetuned/`
- `infer.py` — 在 20 条用例上推理，打印结果，断言无错，写 `results.json`
- `README.md` — 运行说明（训练/推理命令）、模型特定说明
- `requirements.txt` — 额外 pip 依赖（sentence-transformers/FlagEmbedding/numpy<2 等）

## 4. 测试用例格式（20 条）
- **Embedding 模型**：20 条 `{query, positive, negative}` 三元组（用于对比学习训练与检索评测）。
  - 语言域：bge-large-zh → 中文；jina-v3 → 多语种(英/中/西/法/德…)；Qwen3-Embedding → 多语种。
  - 构造来源：参考 README 示例 + 常识，确保 positive 与 query 相关、negative 不相关。
- **Reranker 模型**：20 条 `{query, document, label}`，label∈{0,1}（一半相关一半不相关）。
  - bge-reranker → 中英文；Qwen3-Reranker → 多语种。
- 用例要有多样性（不同主题：地理、物理、历史、编程、生活等），覆盖模型支持的中英多语。

## 5. train.py 要求
- 从 `MODEL_PATH` 加载模型。
- 极小规模微调：1–2 epoch，batch 4–8，lr 2e-5，**max_length 训练时统一用 512**（即使模型支持 8192，训练提速）。
- 保存到本模型目录下 `./finetuned/`。
- **必须无错完成并打印 `TRAIN_OK`**。
- 实现方式（按模型选择最稳妥的，README 对齐）：
  - **Embedding**：`SentenceTransformer` + `SentenceTransformerTrainer` + `MultipleNegativesRankingLoss`（query/positive 二元组）。jina 设置 `default_task`。
  - **bge-reranker-large**：`CrossEncoder` + `CrossEncoderTrainer` + `BinaryCrossEntropyLoss`（AutoModelForSequenceClassification）。
  - **Qwen3-Reranker-0.6B**：transformers 手动循环，`AutoModelForCausalLM`，按 README 的 prefix/suffix + `token_true_id`(yes)/`token_false_id`(no)，取最后 token 的 logits 做 `[false,true]` 的 cross_entropy vs label，backward+step。（避免 ST CrossEncoder 与 causal LM 不匹配。）
- 若 ST Trainer API 不确定，可写**手动 torch 训练循环**（forward→loss→backward→step），只要在 cuda 上能无错训练即可。

## 6. infer.py 要求
- 加载模型：若 `./finetuned/` 存在则优先用，否则用 `MODEL_PATH` 基模型。
- 在 20 条用例上推理：
  - **Embedding**：encode queries + docs，算 cosine 相似矩阵，打印，**断言 embedding 维度**符合 README（Qwen3 1024；bge-large-zh 1024；jina 1024）。
  - **Reranker**：对 20 条 (q,d) 打分，打印分数，**断言得到 20 个分数**。
- 写 `results.json`。
- 打印 `INFER_OK` 并 `exit 0`。不得抛异常。

## 7. 各模型特定要点（来自 README）
| 模型 | 类型 | pooling | 指令 | max_len | dim | 关键 |
|------|------|---------|------|---------|-----|------|
| Qwen3-Embedding-0.6B | embed | last-token | `Instruct: {task}\nQuery:{query}` | 8192(训512) | 1024 | `prompt_name="query"`，padding_side=left，MRL |
| bge-large-zh-v1.5 | embed | CLS | `为这个句子生成表示以用于检索相关文章：` | 512 | 1024 | `normalize_embeddings=True` |
| jina-embeddings-v3 | embed | mean | task 参数 | 8192(训512) | 1024 | `trust_remote_code=True`, `task`/`prompt_name`, Matryoshka `truncate_dim`, **numpy<2** |
| Qwen3-Reranker-0.6B | ranker | —(yes/no logits) | `<Instruct>/<Query>/<Document>` | 8192(训512) | — | `AutoModelForCausalLM`, token yes/no, padding_side=left |
| bge-reranker-large | ranker | —(1 logit) | 无 | 512 | — | `AutoModelForSequenceClassification`, `.logits.view(-1)` |

## 8. 依赖（容器内安装）
- `sentence-transformers>=3.0`（支持 Qwen3 与 jina v3）
- `FlagEmbedding`（bge 的 FlagModel/FlagReranker 备选）
- `transformers>=4.51`（Qwen3）
- `numpy<2`（jina v3 要求；但与其他包可能冲突，必要时仅 jina 子环境使用）

## 9. 验收标准
- 5 个模型 `train.py` 均无错运行并打印 `TRAIN_OK`、产出 `finetuned/`。
- 5 个模型 `infer.py` 均无错运行并打印 `INFER_OK`、产出 `results.json`。
- 训练与推理**计算发生在 PPU(cuda)** 上，非 CPU。
