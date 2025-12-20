import argparse
import torch
import nltk
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import random
import numpy as np

# Custom modules
from utils import TxtLogger, greedy_decode, batch_beam_search_decode, MetricCalculator
from nlp_dataset import NMTDataset, collate_fn, PAD_IDX
from Transformer import TransformerNMT

# ==========================================
# 0. Environment Setup
# ==========================================
nltk_data_path = "/mnt/afs/250010063/AP0004_Midterm/NMT_Project/nltk_data"

if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt ...")
    nltk.download('punkt', download_dir=nltk_data_path)


# ==========================================
# 1. Argument Configuration
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description='Transformer NMT Test Script')

    # --- Data Path Arguments ---
    parser.add_argument('--data_root', type=str, default=r'/mnt/afs/250010063/AP0004_Midterm/NMT_Project/data',
                        help='JSONL root directory')
    parser.add_argument('--save_path', type=str, default=r'/mnt/afs/250010063/AP0004_Midterm/results',
                        help='Path to save results')
    parser.add_argument('--exp', type=str, default=r'Transformer_large',
                        help='Experiment name (folder name)')

    # --- Model Architecture Arguments ---
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of encoder/decoder layers')
    parser.add_argument('--ffn_dim', type=int, default=2048, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.33, help='Dropout rate')
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'],
                        help='Normalization type')
    parser.add_argument('--pos_type', type=str, default='learnable', choices=['sinusoidal', 'learnable'],
                        help='Position embedding type')

    # --- Evaluation Parameters ---
    parser.add_argument('--resume', type=str,
                        default=r'/mnt/afs/250010063/AP0004_Midterm/NMT_Project/results/Transformer_base/best_model.pth',
                        help='Path to checkpoint to evaluate')
    parser.add_argument('--batch_size', type=int, default=120, help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # --- Decoding Strategy Arguments (NEW) ---
    parser.add_argument('--decode', type=str, default='beam', choices=['greedy', 'beam'],
                        help='Decoding strategy to use')
    parser.add_argument('--beam_width', type=int, default=5, 
                        help='Beam width (only used if decode mode is beam)')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ==========================================
# 2. Helper Functions (Masking)
# ==========================================
def generate_square_subsequent_mask(sz, device):
    """Generate a look-ahead mask to prevent attending to future tokens."""
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, trg, device):
    """Create padding masks for encoder/decoder and look-ahead mask for decoder."""
    trg_seq_len = trg.shape[1]

    trg_mask = generate_square_subsequent_mask(trg_seq_len, device)
    src_padding_mask = (src == PAD_IDX)
    trg_padding_mask = (trg == PAD_IDX)
    
    return src_padding_mask, trg_padding_mask, trg_mask


# ==========================================
# 3. Unified Evaluation Logic
# ==========================================
def evaluate(model, dataloader, criterion, device, trg_vocab, metric_calc, decode_method='greedy', beam_width=5):
    """
    Unified evaluation function for both Greedy and Beam Search decoding.
    """
    model.eval()
    epoch_loss = 0
    
    all_src_text = [] 
    all_preds = []    
    all_refs = []     
    
    SOS_IDX = 1
    EOS_IDX = 2
    PAD_IDX = 0 
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(dataloader):
            src, trg = src.to(device), trg.to(device)
            
            # --- 1. Loss Calculation (Common for both) ---
            trg_input = trg[:, :-1]
            trg_label = trg[:, 1:]
            src_padding_mask, trg_padding_mask, trg_mask = create_mask(src, trg_input, device)
            
            output = model(src, trg_input, trg_mask=trg_mask, src_key_padding_mask=src_padding_mask, trg_key_padding_mask=trg_padding_mask)
            
            output_flat = output.contiguous().view(-1, output.shape[-1])
            trg_label_flat = trg_label.contiguous().view(-1)
            epoch_loss += criterion(output_flat, trg_label_flat).item()
            
            # --- 2. Decoding Strategy Switch ---
            src_mask = (src == PAD_IDX).to(device)
            max_len = trg.shape[1] + 20
            
            if decode_method == 'greedy':
                decoded_batch = greedy_decode(
                    model, src, src_mask, max_len, SOS_IDX, EOS_IDX, device
                )
            else:
                decoded_batch = batch_beam_search_decode(
                    model, src, src_mask, max_len, SOS_IDX, EOS_IDX, device, beam_width=beam_width
                )
            
            # --- 3. Process Text (Post-processing) ---
            batch_pred_ids = decoded_batch.tolist()
            batch_trg_ids = trg.tolist()
            
            for pred_ids, trg_ids in zip(batch_pred_ids, batch_trg_ids):
                # Clean IDs: remove SOS, PAD, and cut at EOS
                clean_pred = []
                for idx in pred_ids:
                    if idx == EOS_IDX: break
                    if idx not in [SOS_IDX, PAD_IDX]:
                        clean_pred.append(idx)
                
                clean_trg = []
                for idx in trg_ids:
                    if idx == EOS_IDX: break
                    if idx not in [SOS_IDX, PAD_IDX]:
                        clean_trg.append(idx)
                
                # Convert IDs to Words
                pred_str = " ".join([trg_vocab.idx2word.get(idx, '<unk>') for idx in clean_pred])
                ref_str = " ".join([trg_vocab.idx2word.get(idx, '<unk>') for idx in clean_trg])
                
                # Detokenization (Fix punctuation spacing)
                rep_rules = [
                    (" .", "."), (" ,", ","), (" ?", "?"), (" !", "!"), 
                    (" '", "'"), (" n't", "n't"), (" 's", "'s"), 
                    (" 'm", "'m"), (" 're", "'re"), (" 've", "'ve"),
                    (" 'll", "'ll"), (" 'd", "'d")
                ]
                for old, new in rep_rules:
                    pred_str = pred_str.replace(old, new)
                    ref_str = ref_str.replace(old, new)
                
                all_preds.append(pred_str)
                all_refs.append(ref_str)
                all_src_text.append("") # Placeholder if raw source not available

    # --- 4. Compute Metrics ---
    avg_loss = epoch_loss / len(dataloader)
    ppl = metric_calc.compute_ppl(avg_loss)
    bleu = metric_calc.compute_bleu(all_preds, all_refs)
    bertscore = metric_calc.compute_bertscore(all_preds, all_refs)

    return avg_loss, ppl, bleu, bertscore


# ==========================================
# 4. Main Function
# ==========================================
def main():
    # 1. Parse arguments & Setup
    args = get_args()
    os.makedirs(os.path.join(args.save_path, args.exp), exist_ok=True)
    logger = TxtLogger(save_dir=os.path.join(args.save_path, args.exp))
    logger.log(f"Configurations: {args}")

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")
    
    # 2. Load Vocabulary
    vocap_pth = os.path.join(args.data_root, 'v1', 'vocab.pth')
    print("Loading vocabulary from file...")
    vocap = torch.load(vocap_pth, map_location=device, weights_only=False)
    src_vocab = vocap['src_vocab']
    trg_vocab = vocap['trg_vocab']
    
    # 3. Create Dataset and DataLoader
    test_cache = os.path.join(args.data_root, 'v1', 'test.pt')
    test_dataset = NMTDataset(
        file_path=os.path.join(args.data_root, 'test.jsonl'),
        raw_src=None, 
        raw_trg=None,
        src_vocab=src_vocab, 
        trg_vocab=trg_vocab,
        max_len=1000,
        cache_name=test_cache
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    # 4. Initialize Model
    model = TransformerNMT(
        src_vocab_size=src_vocab.n_words,
        trg_vocab_size=trg_vocab.n_words,
        d_model=args.d_model,
        nhead=args.n_head,
        num_layers=args.n_layers,
        pos_embed_type=args.pos_type,
        norm_type=args.norm_type
    ).to(device)

    # 5. Load Checkpoint and Evaluate
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    if args.resume != '':
        logger.log(f"Loading checkpoint: {args.resume}")
        weights = torch.load(args.resume, map_location=device)
        model.load_state_dict(weights['model'])
        
        metric_calc = MetricCalculator()
        
        logger.log(f"Starting Evaluation using {args.decode} search...")
        
        loss, ppl, bleu, bertscore = evaluate(
            model, 
            test_loader, 
            criterion, 
            device, 
            trg_vocab, 
            metric_calc,
            decode_method=args.decode,
            beam_width=args.beam_width
        )
        
        logger.log(
            f"Test loss: {loss:.4f}, Test bleu: {bleu:.4f}, Test ppl: {ppl:3.2f}, Test bert score: {bertscore:3.2f}%"
        )
    else:
        logger.log("No checkpoint provided via --resume, skipping evaluation.")


if __name__ == '__main__':
    main()