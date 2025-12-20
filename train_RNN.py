import argparse
import time
import math
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sacrebleu
from timm.scheduler import CosineLRScheduler

# Custom modules
from utils import TxtLogger, greedy_decode_rnn
from nlp_dataset import NMTDataset, collate_fn, PAD_IDX, tokenize_cn, tokenize_en, load_data_from_json, Vocab
from RNN import EncoderRNN, LuongAttnDecoderRNN, Seq2Seq, SOS_TOKEN, EOS_TOKEN


# ==========================================
# 1. Argument Configuration
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description='RNN Seq2Seq NMT Training Script')

    # --- Data Path Arguments ---
    parser.add_argument('--data_root', type=str, default=r'/mnt/afs/250010063/AP0004_Midterm/data',
                        help='Root directory for data')
    parser.add_argument('--save_path', type=str, default=r'/mnt/afs/250010063/AP0004_Midterm/results',
                        help='Directory to save results')
    parser.add_argument('--exp', type=str, default=r'RNN_Luong_TF0',
                        help='Experiment name')

    # --- Model Architecture Arguments (RNN Specific) ---
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Hidden size of GRU')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--attn_method', type=str, default='general', choices=['dot', 'general', 'concat'],
                        help='Luong attention method')

    # --- Training Hyperparameters ---
    parser.add_argument('--resume', type=str, default=r'',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=240,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=8e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.0,
                        help='Probability of using teacher forcing during training')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ==========================================
# 2. Helpers & Evaluation Functions
# ==========================================
def _ids_to_sentence(ids, idx2word, pad_id, sos_id, eos_id, unk_token="<unk>"):
    """Convert a list of IDs back to a string sentence."""
    toks = []
    for x in ids:
        if x == eos_id:
            break
        if x in (pad_id, sos_id):
            continue
        toks.append(idx2word.get(int(x), unk_token))
    
    s = " ".join(toks)

    # Detokenization rules
    rep_rules = [
        (" .", "."), (" ,", ","), (" ?", "?"), (" !", "!"),
        (" '", "'"), (" n't", "n't"), (" 's", "'s"),
        (" 'm", "'m"), (" 're", "'re"), (" 've", "'ve"),
        (" 'll", "'ll"), (" 'd", "'d"),
    ]
    for old, new in rep_rules:
        s = s.replace(old, new)
    return s


def evaluate_rnn(model, dataloader, criterion, device, trg_vocab, pad_id, sos_id, eos_id, max_len=200):
    """Evaluate the model using BLEU score and Loss."""
    model.eval()
    total_loss = 0.0
    preds_str = []
    refs_str = []

    with torch.no_grad():
        for (src, trg) in dataloader:
            # Transpose to [seq_len, batch_size]
            src = src.transpose(0, 1).to(device)
            trg = trg.transpose(0, 1).to(device)

            # --- Part 1: Loss (using Teacher Forcing = 1.0) ---
            out_tf = model(src, trg, teacher_forcing_ratio=1.0)
            vocab_size = out_tf.shape[-1]
            
            # Flatten outputs and targets
            out_flat = out_tf[1:].reshape(-1, vocab_size)
            trg_flat = trg[1:].reshape(-1)
            
            loss = criterion(out_flat, trg_flat)
            total_loss += loss.item()

            # --- Part 2: Generation for BLEU (Greedy Decode) ---
            pred_ids = greedy_decode_rnn(
                model=model,
                src=src,
                max_len=max_len,
                pad_id=pad_id,
                sos_id=sos_id,
                eos_id=eos_id,
                device=device,
            )

            # Transpose target back to [batch, seq_len] for comparison
            trg_bt = trg.transpose(0, 1)

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
    bleu = sacrebleu.corpus_bleu(preds_str, [refs_str]).score
    return avg_loss, bleu


def train_epoch(epoch, model, iterator, optimizer, criterion, clip, device, tf_ratio, logger):
    """Training loop for one epoch."""
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        # [seq_len, batch_size]
        src = src.transpose(0, 1).to(device)
        trg = trg.transpose(0, 1).to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg, teacher_forcing_ratio=tf_ratio)
        output_dim = output.shape[-1]
        
        # NOTE: Used reshape instead of view to handle non-contiguous tensors
        output_flat = output[1:].reshape(-1, output_dim)
        trg_flat = trg[1:].reshape(-1)
        
        loss = criterion(output_flat, trg_flat)
        loss.backward()
        
        # Clip gradients to prevent explosion (common in RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        logger.log(f"Epoch{epoch}, Iter [{i+1}/{len(iterator)}] | Loss: {loss.item():.4f}")

    return epoch_loss / len(iterator)


# ==========================================
# 3. Main Execution
# ==========================================
def main():
    # 1. Parse arguments and setup
    args = get_args()
    os.makedirs(os.path.join(args.save_path, args.exp), exist_ok=True)
    logger = TxtLogger(save_dir=os.path.join(args.save_path, args.exp))
    logger.log(f"Configurations: {args}")

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")
    
    # 2. Dataset Preparation
    raw_src_train, raw_trg_train = load_data_from_json(os.path.join(args.data_root, 'train_100k.jsonl'))
    vocap_pth = os.path.join(args.data_root, 'v1', 'vocab.pth')

    # Load or Build Vocabulary
    if os.path.exists(vocap_pth):
        print("Found existing vocabulary file, loading...")
        vocap = torch.load(vocap_pth, map_location=device, weights_only=False)
        src_vocab = vocap['src_vocab']
        trg_vocab = vocap['trg_vocab']
    else:
        print("Building Vocabularies from Training Data...")
        all_src_tokens = []
        all_trg_tokens = []
        for s in raw_src_train: all_src_tokens.extend(tokenize_cn(s))
        for s in raw_trg_train: all_trg_tokens.extend(tokenize_en(s))
        
        src_vocab = Vocab("Chinese", all_src_tokens, min_freq=5)
        trg_vocab = Vocab("English", all_trg_tokens, min_freq=5)
        save_dict = {'src_vocab': src_vocab, 'trg_vocab': trg_vocab}
        torch.save(save_dict, vocap_pth)

    # Define Cache Paths
    train_cache = os.path.join(args.data_root, 'v1', 'train_100k_cache.pt')
    val_cache = os.path.join(args.data_root, 'v1', 'val.pt')
    test_cache = os.path.join(args.data_root, 'v1', 'test.pt')

    # Initialize Datasets
    if os.path.exists(train_cache):
        train_dataset = NMTDataset(None, None, None, src_vocab, trg_vocab, max_len=60, cache_name=train_cache)
    else:
        train_dataset = NMTDataset(None, raw_src_train, raw_trg_train, src_vocab, trg_vocab, max_len=60, cache_name=train_cache)
    
    val_dataset = NMTDataset(
        file_path=os.path.join(args.data_root, 'valid.jsonl'),
        raw_src=None, raw_trg=None, src_vocab=src_vocab, trg_vocab=trg_vocab,
        max_len=60, cache_name=val_cache
    )
    
    test_dataset = NMTDataset(
        file_path=os.path.join(args.data_root, 'test.jsonl'),
        raw_src=None, raw_trg=None, src_vocab=src_vocab, trg_vocab=trg_vocab,
        max_len=60, cache_name=test_cache
    )

    # Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Model Initialization
    encoder = EncoderRNN(
        input_size=src_vocab.n_words,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    decoder = LuongAttnDecoderRNN(
        attn_method=args.attn_method,
        hidden_size=args.hidden_size,
        output_size=trg_vocab.n_words,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Weight Initialization
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
    model.apply(init_weights)

    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 4. Optimizer, Scheduler, Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs, warmup_t=5, warmup_lr_init=3e-4, cycle_limit=1
    )
    criterion = nn.NLLLoss(ignore_index=PAD_IDX)

    # 5. Training Loop Setup
    best_blue = 0
    current_epoch = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_PPL': [], 'val_blue': [], 'lr': []}

    # Resume from Checkpoint
    if args.resume != '':
        logger.log(f"Resuming from checkpoint: {args.resume}")
        weights = torch.load(args.resume, map_location=device)
        model.load_state_dict(weights['model'])
        scheduler.load_state_dict(weights['scheduler'])
        optimizer.load_state_dict(weights['optimizer'])
        current_epoch = weights['epoch'] + 1
        best_blue = weights['performance']
        history = weights['history']
        
        # Initial test on resume
        test_loss, test_blue = evaluate_rnn(
            model, test_loader, criterion, device, trg_vocab, PAD_IDX, SOS_TOKEN, EOS_TOKEN, max_len=200
        )
        logger.log(f"Resumed Test Blue Score: {test_blue:.2f}%")

    # Start Training
    for epoch in range(current_epoch, args.epochs):
        train_loss = train_epoch(epoch, model, train_loader, optimizer, criterion, args.clip, device, args.teacher_forcing_ratio, logger)
        
        val_loss, val_blue = evaluate_rnn(
            model, val_loader, criterion, device, trg_vocab, PAD_IDX, SOS_TOKEN, EOS_TOKEN, max_len=200
        )
        
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]['lr']

        # Update History
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_PPL'].append(math.exp(min(train_loss, 100))) 
        history['val_blue'].append(val_blue)
        history['lr'].append(lr)
        
        logger.log(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f} | Train PPL: {math.exp(min(train_loss, 100)):7.3f} | Val Loss: {val_loss:.3f} | Val Blue Score: {val_blue:.3f}')
        
        # Save Best Model
        if val_blue > best_blue:
            best_blue = val_blue
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'performance': best_blue,
                'epoch': epoch,
                'history': history,
            }
            torch.save(save_dict, os.path.join(args.save_path, args.exp, 'best_model.pth'))
            logger.log(f"Best model saved.")

    # 6. Finalize
    pd.DataFrame(history).to_csv(os.path.join(args.save_path, args.exp, 'training_history.csv'), index=False)
    logger.log(f"Training history saved")

    # Final Test
    weights = torch.load(os.path.join(args.save_path, args.exp, 'best_model.pth'), map_location=device)
    model.load_state_dict(weights['model'])
    
    test_loss, test_blue = evaluate_rnn(
        model, test_loader, criterion, device, trg_vocab, PAD_IDX, SOS_TOKEN, EOS_TOKEN, max_len=200
    )
    logger.log(f"Final Test Blue Score: {test_blue:.2f}%")


if __name__ == '__main__':
    main()