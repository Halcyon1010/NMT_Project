# 中英机器翻译 (NMT) 项目

这是一个基于 PyTorch 实现的 **中文到英文 (Zh-En)** 神经机器翻译项目。项目涵盖了RNN 、Transformer 以及预训练模型微调 (NLLB) 的完整流程，旨在对比不同架构在翻译任务上的表现。

## 📋 项目简介

本项目实现了三种主流的 NMT 解决方案，并提供了统一的数据处理、训练、推理和评估接口：

1.  **RNN (Seq2Seq)**:
    * 基于 **GRU** 的编码器-解码器架构。
    * 实现了 **Luong Attention** 机制 (支持 `dot`, `general`, `concat` 三种实现方式)。
    * 支持 Teacher Forcing 训练策略。
2.  **Transformer**:
    * 基于 *Attention Is All You Need* 的标准架构。
    * **消融实验支持**：支持切换归一化层 (`LayerNorm` vs `RMSNorm`) 和位置编码 (`Sinusoidal` vs `Learnable`)。
3.  **NLLB (Fine-tuning)**:
    * 基于 Meta 的 **No Language Left Behind (NLLB)** 模型 (`facebook/nllb-200-distilled-600M`) 进行微调。
    * 使用 Hugging Face `transformers` 库实现。


## ⚙️ 环境依赖
```bash
conda create -n NMT python=3.10.0
conda activate NMT

pip install -r requirements.txt
```
注意：首次运行代码时，脚本会自动下载 NLTK 的 punkt 分词数据包到 `nltk_data/` 目录。


## 📊 数据准备

项目主要支持 **JSONL 格式** 的数据集。请在 `data/` 目录下准备以下文件：

- `train_100k.jsonl`（训练集）
- `valid.jsonl`（验证集）
- `test.jsonl`（测试集）

---

### 数据格式说明

每一行包含一个 **JSON 对象**，必须包含以下字段：

- `zh`：中文源句  
- `en`：英文目标句  

---

### 数据格式示例

```json
{"zh": "我爱自然语言处理。", "en": "I love Natural Language Processing."}
{"zh": "深度学习正在改变世界。", "en": "Deep learning is changing the world."}
```

## 🚀 快速开始（Quick Start）

**注意：** 以下命令中的路径（如 `--data_root`）请根据您实际的文件位置进行调整。

---

### 1. 训练 RNN 模型（Seq2Seq + Attention）

训练一个基于 **Luong Attention（General）** 的双层 GRU 模型：

```bash
python train_RNN.py \
  --data_root ./data \
  --save_path ./results \
  --exp RNN_Luong_general \
  --hidden_size 512 \
  --n_layers 2 \
  --attn_method general \
  --batch_size 240 \
  --epochs 50 \
  --lr 8e-4
```

### 2. 训练 Transformer 模型

训练一个标准 **Transformer** 模型（使用 Learnable Positional Encoding 和 LayerNorm）：

```bash
python train_transformer.py \
    --data_root ./data \
    --save_path ./results \
    --exp Transformer_base \
    --d_model 512 \
    --n_head 8 \
    --n_layers 6 \
    --ffn_dim 2048 \
    --norm_type layernorm \
    --pos_type learnable \
    --batch_size 120 \
    --epochs 50
```

### 3. 微调 NLLB 模型

使用 Hugging Face `transformers` 微调 `facebook/nllb-200-distilled-600M`

```bash
python Finetune_NLLB.py \
    --data_root ./data \
    --output_dir ./results/nllb_finetuned \
    --model_name_or_path facebook/nllb-200-distilled-600M \
    --train_file train_100k.jsonl \
    --valid_file valid.jsonl \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-4
```
**提示**：如果是显存较小，请减小 `per_device_train_batch_size` 并增大 `gradient_accumulation_steps`。

## 🔍 推理与评估 (Inference & Evaluation)
### 统一评估脚本 (`inference.py`)

这是推荐的评估方式，可以同时加载多个模型并在测试集上计算 BLEU 和 BERTScore。
```bash
python inference.py \
    --data_root ./data \
    --test_file test.jsonl \
    --save_path ./inference_results \
    --models rnn transformer nllb \
    --rnn_ckpt ./results/RNN_Luong_general/best_model.pth \
    --trans_ckpt ./results/Transformer_base/best_model.pth \
    --nllb_path ./results/nllb_finetuned/final \
    --batch_size 100 \
    --beam_width 5 \
    --device cuda
```

### 单模型测试
如果您只想测试单个模型，可以使用对应的测试脚本：

- **RNN:**
    ```bash
    python test_RNN.py \
    --resume ./results/RNN_Luong_general/best_model.pth \
    --decode beam \
    --beam_width 5
    ```

- **Transformer:**
    ```bash
    python test_transformer.py \
    --resume ./results/Transformer_base/best_model.pth \
    --decode beam \
    --beam_width 5
    ```

- **NLLB:**
    ```bash
    python test_NLLB.py \
    --model_path ./results/nllb_finetuned/final
    ```

## 🛠 技术细节

- **文本处理：**
  - **中文：** 使用 `jieba` 进行分词。
  - **英文：** 使用 `nltk.word_tokenize` 进行分词。
  - **词表：** 自动构建并保存为 `.pth` 文件，支持 `<pad>`、`<sos>`、`<eos>`、`<unk>` 等特殊 Token。

- **解码策略：**
  - 支持 **Greedy Search**（贪婪搜索）。
  - 支持 **Beam Search**（束搜索），包含长度惩罚（Length Penalty）机制，支持 Batch 并行解码以提高推理速度。

- **评价指标：**
  - **BLEU：** 使用 `sacrebleu` 库计算语料库级别的 BLEU 分数。
  - **BERTScore：** 利用预训练 BERT 模型计算生成的语义相似度。
  - **PPL（Perplexity）：** 用于衡量语言模型的预测能力。


## 📂 目录结构

```text
NMT_Project/
├── data/                   # 数据集目录 (JSONL 格式)
│   ├── train_100k.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── nltk_data/              # NLTK 数据缓存
├── results/                # 实验结果与模型权重保存目录
├── Finetune_NLLB.py        # NLLB 微调脚本
├── inference.py            # 统一推理与评估脚本
├── nlp_dataset.py          # 数据预处理与 Dataset 定义
├── RNN.py                  # RNN 模型架构定义
├── Transformer.py          # Transformer 模型架构定义
├── train_RNN.py            # RNN 训练脚本
├── train_transformer.py    # Transformer 训练脚本
├── test_RNN.py             # RNN 测试脚本
├── test_transformer.py     # Transformer 测试脚本
├── test_NLLB.py            # NLLB 测试脚本
└── utils.py                # 工具函数 (Beam Search, Metrics, Logger)

