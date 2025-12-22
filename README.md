# Chinese–English Neural Machine Translation (NMT) Project

This project is a **Chinese-to-English (Zh–En) Neural Machine Translation system** implemented with **PyTorch**.  
It provides a comprehensive pipeline covering **RNN-based models, Transformer models, and fine-tuning of pretrained models (NLLB)**, with the goal of systematically comparing different architectures on machine translation tasks.

---

## 📋 Project Overview

This repository implements three mainstream NMT solutions and provides unified interfaces for data preprocessing, training, inference, and evaluation, enabling fair and convenient comparisons across models.

### 1. RNN (Seq2Seq + Attention)
- Encoder–decoder architecture based on **GRU**
- Implements **Luong Attention**, supporting `dot`, `general`, and `concat` variants
- Supports **Teacher Forcing** during training

### 2. Transformer
- Standard Transformer architecture proposed in *Attention Is All You Need*
- **Ablation-friendly design**, supporting:
  - Normalization layers: `LayerNorm` vs. `RMSNorm`
  - Positional encodings: `Sinusoidal` vs. `Learnable`

### 3. NLLB (Fine-tuning)
- Fine-tuning of Meta’s **No Language Left Behind (NLLB)** pretrained model
- Model used: `facebook/nllb-200-distilled-600M`
- Implemented using the Hugging Face `transformers` library

---

## ⚙️ Environment Setup

```bash
conda create -n NMT python=3.10.0
conda activate NMT

pip install -r requirements.txt
```

**Note**:
At the first run, the scripts will automatically download the NLTK `punkt` tokenizer data and cache it in the `nltk_data/` directory.


## 📊 Data Preparation

The project uses datasets in JSONL (JSON Lines) format.
Please prepare the following files under the `data/` directory:

- `train_100k.jsonl` — training set

- `valid.jsonl` — validation set

- `test.jsonl` — test set

### Data Format

Each line corresponds to a single JSON object and must contain the following fields:

- `zh`: source sentence in Chinese

- `en`: target sentence in English

### Example
```json
{"zh": "我爱自然语言处理。", "en": "I love Natural Language Processing."}
{"zh": "深度学习正在改变世界。", "en": "Deep learning is changing the world."}
```

## 🚀 Quick Start

**Note**:
Please adjust path-related arguments (e.g., `--data_root`) according to your local directory structure.

### 1. Train an RNN Model (Seq2Seq + Attention)

Train a two-layer GRU model with Luong Attention (General):
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

### 2. Train a Transformer Model

Train a standard Transformer model using learnable positional encoding and LayerNorm:
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
### 3. Fine-tune the NLLB Model

Fine-tune `facebook/nllb-200-distilled-600M` using Hugging Face `transformers`:
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
**Tip**:
If GPU memory is limited, reduce `per_device_train_batch_size` and increase `gradient_accumulation_steps` accordingly.


## 🔍 Inference & Evaluation
### Unified Evaluation Script (`inference.py`)

This is the recommended evaluation entry point.
It supports loading multiple models simultaneously and computes **BLEU** and **BERTScore** on the test set.

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

### Single-Model Testing

If you only want to evaluate a single model, use the corresponding test script:

**RNN**
```bash
python test_RNN.py \
  --resume ./results/RNN_Luong_general/best_model.pth \
  --decode beam \
  --beam_width 5
```

**Transformer**
```bash
python test_transformer.py \
  --resume ./results/Transformer_base/best_model.pth \
  --decode beam \
  --beam_width 5

```

**NLLB**
```bash
python test_NLLB.py \
  --model_path ./results/nllb_finetuned/final
```

## 🛠 Technical Details
### Text Processing

Chinese tokenization: `jieba`

English tokenization: `nltk.word_tokenize`

Vocabulary: Automatically built and saved as `.pth` files, supporting special tokens
`<pad>`, `<sos>`, `<eos>`, `<unk>`

### Decoding Strategies

Greedy Search

Beam Search, with length penalty support and batch-parallel decoding for faster inference

### Evaluation Metrics

**BLEU**: Corpus-level BLEU score computed with `sacrebleu`

**BERTScore**: Semantic similarity computed using a pretrained BERT model

**PPL (Perplexity)**: Measures the predictive uncertainty of the language model

## 📂 Project Structure

```text
NMT_Project/
├── data/                   # Datasets (JSONL format)
│   ├── train_100k.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── nltk_data/              # Cached NLTK resources
├── results/                # Saved models and  experimental outputs
├── Finetune_NLLB.py        # NLLB fine-tuning script
├── inference.py            # Unified inference & evaluation script
├── nlp_dataset.py          # Data preprocessing and Dataset definitions
├── RNN.py                  # RNN model definitions
├── Transformer.py          # Transformer model definitions
├── train_RNN.py            # RNN training script
├── train_transformer.py    # Transformer training script
├── test_RNN.py             # RNN testing script
├── test_transformer.py     # Transformer testing script
├── test_NLLB.py            # NLLB testing script
└── utils.py                # Utilities (Beam Search, metrics, logging)
```