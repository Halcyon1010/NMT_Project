import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import pandas as pd
from test_RNN import evaluate_rnn_metrics
from test_transformer import evaluate as evaluate_transformer
from utils import MetricCalculator, batch_beam_search_decode, greedy_decode, greedy_decode_rnn, beam_decode_rnn
from nlp_dataset import NMTDataset, collate_fn, PAD_IDX, SOS_IDX, EOS_IDX
from RNN import EncoderRNN, LuongAttnDecoderRNN, Seq2Seq
from Transformer import TransformerNMT
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings("ignore")

pad_id, sos_id, eos_id = 0, 1, 2
def get_args():
    parser = argparse.ArgumentParser(description='Unified Inference Script for RNN, Transformer, and NLLB')

    # Data & Paths
    parser.add_argument('--data_root', type=str, default='/mnt/afs/250010063/AP0004_Midterm/NMT_Project/data', help='Data directory')
    parser.add_argument('--test_file', type=str, default='test.jsonl', help='Test file name')
    parser.add_argument('--save_path', type=str, default='./inference_results', help='Path to save results')
    
    # Model Selection
    parser.add_argument('--models', nargs='+', default=['rnn', 'transformer', 'nllb'], 
                        choices=['rnn', 'transformer', 'nllb'], help='Models to evaluate')

    # Checkpoints
    parser.add_argument('--rnn_ckpt', type=str, default='/mnt/afs/250010063/AP0004_Midterm/NMT_Project/results/RNN_Luong_general/best_model.pth')
    parser.add_argument('--trans_ckpt', type=str, default='/mnt/afs/250010063/AP0004_Midterm/NMT_Project/results/Transformer_lr_2e3/best_model.pth')
    parser.add_argument('--nllb_path', type=str, default='/mnt/afs/250010063/AP0004_Midterm/NMT_Project/results/nllb_600m_zh2en/final')

    # Model Hyperparameters (Must match training)
    # RNN
    parser.add_argument('--rnn_hidden', type=int, default=512)
    parser.add_argument('--rnn_layers', type=int, default=2)
    parser.add_argument('--rnn_attn', type=str, default='general')
    
    # Transformer
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8) # or 16
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--pos_type', type=str, default='learnable')
    parser.add_argument('--norm_type', type=str, default='layernorm')

    # Inference Settings
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()

def load_vocab(data_root, device):
    vocab_path = os.path.join(data_root, 'v1', 'vocab.pth')
    print(f"Loading vocab from {vocab_path}...")
    vocab = torch.load(vocab_path, map_location=device, weights_only=False)
    return vocab['src_vocab'], vocab['trg_vocab']

def detokenize(ids, vocab):
    """Convert IDs to sentence string with basic cleanup."""
    tokens = [vocab.idx2word.get(idx, '<unk>') for idx in ids if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]]
    sent = " ".join(tokens)
    # Basic detokenization rules
    rep_rules = [(" .", "."), (" ,", ","), (" ?", "?"), (" !", "!"), (" '", "'"), (" n't", "n't"), (" 's", "'s")]
    for old, new in rep_rules:
        sent = sent.replace(old, new)
    return sent

def print_samples(model_name, sources, references, predictions, n=3):
    print(f"\n{'='*20} {model_name} Sample Translations {'='*20}")
    for i in range(min(n, len(sources))):
        print(f"Src : {sources[i]}")
        print(f"Ref : {references[i]}")
        print(f"Pred: {predictions[i]}")
        print("-" * 50)


# ==========================================
# 3. Model Evaluators
# ==========================================

def evaluate_custom_model(model_name, model, dataloader, trg_vocab, device, args, calc):
    """Shared logic for RNN and Transformer."""
    model.eval()
    preds_str = []
    refs_str = []
    srcs_str = [] # Placeholder or reconstruct if needed

    print(f"Evaluating {model_name}...")
    with torch.no_grad():
        for src, trg in tqdm(dataloader):
            src, trg = src.to(device), trg.to(device)
            
            # RNN expects [Seq, Batch], Transformer expects [Batch, Seq]
            if model_name == 'RNN':
                src = src.transpose(0, 1)
                # trg not needed for inference, but needed for ref
            
            src_mask = (src == PAD_IDX).transpose(0, 1) if model_name == 'RNN' else (src == PAD_IDX)
            max_len = 200

            # Decode
            if model_name == 'RNN':
                # Note: Assuming batch_beam_search_decode handles RNN specifics or using greedy wrapper
                # For RNN, usually we implemented sample-wise beam or greedy.
                # Here simplified to greedy for speed check, or assume beam func handles it.
                # Let's use the provided batch_beam_search_decode if compatible, else greedy.
                # Check your utils for RNN support. Assuming greedy for RNN stability here:
                decoded = beam_decode_rnn(
                    model=model,
                    src=src,
                    beam_width=args.beam_width,
                    max_len=max_len,
                    pad_id=pad_id,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    device=device,
                    length_penalty_alpha=0.6,
                )
                
                # Transpose back for decoding loop if output is [Batch, Seq]
                # RNN greedy usually returns [Batch, Seq]
            else:
                # Transformer
                decoded = batch_beam_search_decode(
                    model, src, src_mask, max_len, SOS_IDX, EOS_IDX, device, beam_width=args.beam_width
                )

            # Post-process
            batch_preds = decoded.tolist()
            batch_refs = trg.transpose(0, 1).tolist() if model_name == 'RNN' else trg.tolist()

            for p_ids, r_ids in zip(batch_preds, batch_refs):
                preds_str.append(detokenize(p_ids, trg_vocab))
                refs_str.append(detokenize(r_ids, trg_vocab))
                srcs_str.append("N/A (Raw text not in loader)") 

    # Metrics
    bleu = calc.compute_bleu(preds_str, refs_str)
    bert = calc.compute_bertscore(preds_str, refs_str)
    bleu4 = calc.compute_bleu(preds_str, refs_str)

    return {"BLEU": bleu, "BERTScore": bert, "BLEU4": bleu4}, preds_str, refs_str


def evaluate_nllb(model_path, test_file, device, args, calc):
    """NLLB Evaluation logic."""
    print("Loading NLLB...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="zho_Hans", tgt_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    
    # Load raw text directly
    with open(test_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Simple batching
    batch_size = args.batch_size
    preds_str = []
    refs_str = []
    srcs_str = []

    print("Evaluating NLLB...")
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        src_texts = [x['zh'] for x in batch]
        tgt_texts = [x['en'] for x in batch]
        
        inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        
        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs, 
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=256,
                num_beams=args.beam_width
            )
        
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        preds_str.extend(decoded)
        refs_str.extend(tgt_texts)
        srcs_str.extend(src_texts)

    # Metrics
    bleu = calc.compute_bleu(preds_str, refs_str)
    bert = calc.compute_bertscore(preds_str, refs_str)
    bleu4 = calc.compute_bleu_4(preds_str, refs_str)
    return {"BLEU": bleu, "BERTScore": bert, "BLEU4": bleu4}, preds_str, refs_str, srcs_str


# ==========================================
# 4. Main
# ==========================================
def main():
    args = get_args()
    device = torch.device(args.device)
    calc = MetricCalculator(device=args.device)
    criterion = nn.CrossEntropyLoss()
    results_summary = {}
    max_len = 200
    # --- 1. Load Vocab & Dataset for RNN/Transformer ---
    if 'rnn' in args.models or 'transformer' in args.models:
        src_vocab, trg_vocab = load_vocab(args.data_root, device)
        test_cache = os.path.join(args.data_root, 'v1', 'test.pt')
        
        test_dataset = NMTDataset(
            file_path=os.path.join(args.data_root, args.test_file),
            raw_src=None, raw_trg=None,
            src_vocab=src_vocab, trg_vocab=trg_vocab,
            max_len=1000, cache_name=test_cache
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- 2. Evaluate RNN ---
    if 'rnn' in args.models:
        encoder = EncoderRNN(src_vocab.n_words, args.rnn_hidden, args.rnn_layers).to(device)
        decoder = LuongAttnDecoderRNN(args.rnn_attn, args.rnn_hidden, trg_vocab.n_words, args.rnn_layers).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)
        
        ckpt = torch.load(args.rnn_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        
        metric_calc = MetricCalculator(device=str(device))

        loss, ppl, bleu, bleu4, bertscore = evaluate_rnn_metrics(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            trg_vocab=trg_vocab,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            max_len=max_len,
            metric_calc=metric_calc,
            decode_type='beam',
            beam_width=args.beam_width,
            length_penalty_alpha=0.6,
        )
        print("For RNN")
        print(
            f"BLEU4: {bleu4:.4f}, BLEU: {bleu:.4f}, BERTScore: {bertscore:3.2f}%"
        )

    # --- 3. Evaluate Transformer ---
    if 'transformer' in args.models:
        model = TransformerNMT(
            src_vocab_size=src_vocab.n_words,
            trg_vocab_size=trg_vocab.n_words,
            d_model=args.d_model, nhead=args.n_head, num_layers=args.n_layers,
            pos_embed_type=args.pos_type, norm_type=args.norm_type
        ).to(device)
        metric_calc = MetricCalculator()
        ckpt = torch.load(args.trans_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
        
        loss, ppl, bleu, bleu4, bertscore = evaluate_transformer(
            model, 
            test_loader, 
            criterion, 
            device, 
            trg_vocab, 
            metric_calc,
            decode_method='beam',
            beam_width=args.beam_width
        )
        print("For transformer")
        print(
            f"Test bleu: {bleu:.4f}, Test bleu4: {bleu4:.4f},Test bert score: {bertscore:3.2f}%"
        )

    # --- 4. Evaluate NLLB ---
    if 'nllb' in args.models:
        test_path = os.path.join(args.data_root, args.test_file)
        metrics, preds, refs, srcs = evaluate_nllb(args.nllb_path, test_path, device, args, calc)
        results_summary['NLLB'] = metrics
        print("For NLLB")
        print(
            f"Test bleu: {metrics['BLEU']:.4f}, Test bleu4: {metrics['BLEU4']:.4f},Test bert score: {metrics['BERTScore']:3.2f}%"
        )


if __name__ == "__main__":
    main()
