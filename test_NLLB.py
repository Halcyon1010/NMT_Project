import argparse
import torch
import math
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader, Dataset
import nltk
from utils import MetricCalculator

# ==========================================
# 1. Environment & Setup
# ==========================================
nltk_data_path = "/mnt/afs/250010063/AP0004_Midterm/NMT_Project/nltk_data"

if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt...")
    nltk.download('punkt', download_dir=nltk_data_path)


# ==========================================
# 2. Dataset Definition
# ==========================================
class NMTDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# ==========================================
# 3. Evaluation Logic
# ==========================================
def evaluate(args):
    device = torch.device(args.device)
    
    # --- 1. Load Model & Tokenizer ---
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    model.eval()

    # --- 2. Configure Language Settings ---
    src_lang = "zho_Hans"
    tgt_lang = "eng_Latn"
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    
    # Get Forced BOS Token ID (Crucial for NLLB translation direction)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    print(f"Source Lang: {src_lang}, Target Lang: {tgt_lang}, Forced BOS ID: {forced_bos_token_id}")

    # --- 3. Load Data ---
    dataset = NMTDataset(args.test_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- 4. Initialize Metric Calculator ---
    # Note: MetricCalculator loads COMET/BERTScore which consumes VRAM.
    calc = MetricCalculator(device=args.device)

    all_src = []
    all_refs = []
    all_preds = []
    total_loss = 0
    total_count = 0

    print("Starting evaluation loop...")
    
    # --- 5. Inference Loop ---
    for batch in tqdm(dataloader):
        # Adjust keys based on your jsonl structure (e.g., 'zh', 'en')
        src_texts = batch['zh'] 
        tgt_texts = batch['en']

        # A. Prepare Inputs
        inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        labels = tokenizer(text_target=tgt_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        # B. Calculate PPL (Loss)
        # Set padding labels to -100 to ignore in loss computation
        loss_labels = labels["input_ids"].clone()
        loss_labels[loss_labels == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(**inputs, labels=loss_labels)
            loss = outputs.loss
            total_loss += loss.item() * len(src_texts)
            total_count += len(src_texts)

        # C. Generate Translations (Beam Search)
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=256,
                num_beams=4, 
            )
        
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Collect Results
        all_src.extend(src_texts)
        all_refs.extend(tgt_texts)
        all_preds.extend(decoded_preds)

    # --- 6. Compute Metrics ---
    print("Computing metrics...")
    
    avg_loss = total_loss / total_count
    ppl = calc.compute_ppl(avg_loss)
    bleu = calc.compute_bleu(all_preds, all_refs)
    gleu = calc.compute_gleu(all_preds, all_refs)
    
    # BERTScore and COMET are computationally expensive
    bert = calc.compute_bertscore(all_preds, all_refs)
    comet = calc.compute_comet(all_src, all_preds, all_refs)

    # --- 7. Output & Save Results ---
    results = {
        "PPL": ppl,
        "BLEU": bleu,
        "GLEU": gleu,
        "BERTScore": bert,
        "COMET": comet
    }
    
    print("\n" + "="*30)
    print("Final Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print("="*30)

    # Save detailed predictions for manual inspection
    output_pred_file = os.path.join(args.output_dir, "predictions.jsonl")
    with open(output_pred_file, "w", encoding="utf-8") as f:
        for s, r, p in zip(all_src, all_refs, all_preds):
            json.dump({"src": s, "ref": r, "pred": p}, f, ensure_ascii=False)
            f.write("\n")
    print(f"Predictions saved to {output_pred_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Path to the fine-tuned model checkpoint
    parser.add_argument("--model_path", type=str, default=r'/mnt/afs/250010063/AP0004_Midterm/results/nllb_600m_zh2en/final', 
                        help="Path to the fine-tuned model directory")
    
    # Test dataset path
    parser.add_argument("--test_file", type=str, default="/mnt/afs/250010063/AP0004_Midterm/data/test.jsonl")
    
    # Output directory for results
    parser.add_argument("--output_dir", type=str, default="/mnt/afs/250010063/AP0004_Midterm/results/nllb_600m_zh2en/test")
    
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args)