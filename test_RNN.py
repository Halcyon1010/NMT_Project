import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import nltk
from tqdm import tqdm

# Custom modules (Adjust paths as necessary)
from utils import TxtLogger, MetricCalculator, greedy_decode_rnn, beam_decode_rnn
from nlp_dataset import NMTDataset, collate_fn, PAD_IDX
from RNN import EncoderRNN, LuongAttnDecoderRNN, Seq2Seq

# -------------------------
# NLTK Setup
# -------------------------
nltk_data_path = "/mnt/afs/250010063/AP0004_Midterm/NMT_Project/nltk_data"
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)


# ==========================================
# 1) Arguments
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="RNN NMT Test Metrics (BLEU/GLEU/BERTScore/PPL)")

    # Data and Path arguments
    parser.add_argument("--data_root", type=str, default=r"/mnt/afs/250010063/AP0004_Midterm/NMT_Project/data")
    parser.add_argument("--save_path", type=str, default=r"/mnt/afs/250010063/AP0004_Midterm/results")
    parser.add_argument("--exp", type=str, default="RNN_Luong_general")
    parser.add_argument("--resume", type=str, default=r"/mnt/afs/250010063/AP0004_Midterm/results/RNN_Luong_general/best_model.pth")

    # RNN Architecture (Must match training configuration)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--attn_method", type=str, default="general", choices=["dot", "general", "concat"])

    # Decoding Strategy
    parser.add_argument("--decode", type=str, default="beam", choices=["greedy", "beam"])
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--len_penalty", type=float, default=0.0)

    # Evaluation Settings
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--max_len_extra", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ==========================================
# 2) Helper Functions (Text Processing)
# ==========================================
def detok_fix(s: str) -> str:
    """Fixes common tokenization artifacts in the output string."""
    rep_rules = [
        (" .", "."), (" ,", ","), (" ?", "?"), (" !", "!"),
        (" '", "'"), (" n't", "n't"), (" 's", "'s"),
        (" 'm", "'m"), (" 're", "'re"), (" 've", "'ve"),
        (" 'll", "'ll"), (" 'd", "'d")
    ]
    for old, new in rep_rules:
        s = s.replace(old, new)
    return s


def _ids_to_sentence(ids, idx2word, pad_id, sos_id, eos_id):
    """Converts a list of IDs to a sentence string, handling special tokens."""
    clean = []
    for idx in ids:
        if idx == eos_id:
            break
        if idx not in (pad_id, sos_id):
            clean.append(idx)
    sent = " ".join([idx2word.get(int(i), "<unk>") for i in clean])
    return detok_fix(sent)


# ==========================================
# 3) Evaluation Loop
# ==========================================
def evaluate_rnn_metrics(
    model,
    dataloader,
    criterion,
    device,
    trg_vocab,
    pad_id,
    sos_id,
    eos_id,
    max_len=200,
    decode_type='greedy',
    metric_calc=None,
    beam_width=5,
    length_penalty_alpha=0.0,
):
    model.eval()
    total_loss = 0.0
    preds_str, refs_str = [], []

    with torch.no_grad():
        for (src, trg) in tqdm(dataloader, desc="Evaluating RNN"):
            # Transpose to [T, B] for RNN
            src = src.transpose(0, 1).to(device)
            trg = trg.transpose(0, 1).to(device)

            # --- Part 1: Loss (Teacher Forcing) ---
            out_tf = model(src, trg, teacher_forcing_ratio=1.0)
            vocab_size = out_tf.shape[-1]
            
            # Flatten for loss (skip SOS)
            out_flat = out_tf[1:].reshape(-1, vocab_size)
            trg_flat = trg[1:].reshape(-1)
            loss = criterion(out_flat, trg_flat)
            total_loss += loss.item()

            # --- Part 2: Generation ---
            if decode_type == "greedy":
                pred_ids = greedy_decode_rnn(
                    model=model,
                    src=src,
                    max_len=max_len,
                    pad_id=pad_id,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    device=device,
                )
            else:
                pred_ids = beam_decode_rnn(
                    model=model,
                    src=src,
                    beam_width=beam_width,
                    max_len=max_len,
                    pad_id=pad_id,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    device=device,
                    length_penalty_alpha=length_penalty_alpha,
                )

            # Convert IDs to Text
            trg_bt = trg.transpose(0, 1)  # Back to [B, T]
            B = pred_ids.shape[0]
            for b in range(B):
                pred_sent = _ids_to_sentence(
                    pred_ids[b].tolist(),
                    idx2word=trg_vocab.idx2word,
                    pad_id=pad_id, sos_id=sos_id, eos_id=eos_id
                )
                ref_sent = _ids_to_sentence(
                    trg_bt[b].tolist(),
                    idx2word=trg_vocab.idx2word,
                    pad_id=pad_id, sos_id=sos_id, eos_id=eos_id
                )
                preds_str.append(pred_sent)
                refs_str.append(ref_sent)

    avg_loss = total_loss / max(1, len(dataloader))

    if metric_calc is None:
        metric_calc = MetricCalculator(device=str(device))

    # Compute Metrics
    ppl = metric_calc.compute_ppl(avg_loss)
    bleu = metric_calc.compute_bleu(preds_str, refs_str)
    bertscore = metric_calc.compute_bertscore(preds_str, refs_str)

    return avg_loss, ppl, bleu, bertscore


# ==========================================
# 4) Main
# ==========================================
def main():
    args = get_args()

    # Logger setup
    os.makedirs(os.path.join(args.save_path, args.exp), exist_ok=True)
    logger = TxtLogger(save_dir=os.path.join(args.save_path, args.exp))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")
    logger.log(f"Args: {args}")

    # ---- Load Vocabulary ----
    vocap_pth = os.path.join(args.data_root, "v1", "vocab.pth")
    vocap = torch.load(vocap_pth, map_location=device, weights_only=False)
    src_vocab = vocap["src_vocab"]
    trg_vocab = vocap["trg_vocab"]

    pad_id, sos_id, eos_id = 0, 1, 2

    # ---- Dataset & DataLoader ----
    test_cache = os.path.join(args.data_root, "v1", "test.pt")
    test_dataset = NMTDataset(
        file_path=os.path.join(args.data_root, "test.jsonl"),
        raw_src=None,
        raw_trg=None,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        max_len=1000,
        cache_name=test_cache,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ---- Build Model ----
    encoder = EncoderRNN(
        input_size=src_vocab.n_words,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    decoder = LuongAttnDecoderRNN(
        attn_method=args.attn_method,
        hidden_size=args.hidden_size,
        output_size=trg_vocab.n_words,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = Seq2Seq(encoder, decoder, device).to(device)

    # ---- Criterion ----
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    # ---- Load Checkpoint ----
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # Handle state dict wrapped in 'model' key or direct dictionary
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        logger.log(f"Loaded checkpoint: {args.resume}")

    # ---- Evaluate ----
    metric_calc = MetricCalculator(device=str(device))

    loss, ppl, bleu, bertscore = evaluate_rnn_metrics(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        trg_vocab=trg_vocab,
        pad_id=pad_id,
        sos_id=sos_id,
        eos_id=eos_id,
        max_len=args.max_len,
        metric_calc=metric_calc,
        decode_type=args.decode,
        beam_width=args.beam_width,
        length_penalty_alpha=args.len_penalty,
    )

    logger.log(
        f"RNN Test loss: {loss:.4f}, BLEU: {bleu:.4f}, PPL: {ppl:3.2f}, BERTScore: {bertscore:3.2f}%"
    )

if __name__ == "__main__":
    main()