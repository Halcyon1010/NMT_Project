import os
# Set Torch cache directory (Environment specific)
os.environ["TORCH_HOME"] = "/mnt/afs/250010063/torch_cache"

import argparse
import time
import math
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
from utils import TxtLogger, greedy_decode
from nlp_dataset import NMTDataset, collate_fn, PAD_IDX, tokenize_cn, tokenize_en, load_data_from_json, Vocab
from Transformer import TransformerNMT

# Set Torch Hub directory
torch.hub.set_dir(os.path.join(os.environ["TORCH_HOME"], "hub"))


# ==========================================
# 1. Argument Configuration
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description='Transformer NMT Training Script')

    # --- Data Path Arguments ---
    parser.add_argument('--data_root', type=str, default=r'/mnt/afs/250010063/AP0004_Midterm/data',
                        help='JSONL root directory')
    parser.add_argument('--save_path', type=str, default=r'/mnt/afs/250010063/AP0004_Midterm/results',
                        help='Path to save results')
    parser.add_argument('--exp', type=str, default=r'Transformer_large',
                        help='Experiment name (folder name)')

    # --- Model Architecture Arguments ---
    parser.add_argument('--d_model', type=int, default=1024, help='Model dimension')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=8, help='Number of encoder/decoder layers')
    parser.add_argument('--ffn_dim', type=int, default=4096, help='FFN dimension')
    parser.add_argument('--dropout', type=float, default=0.33, help='Dropout rate')
    
    # Ablation Parameters
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'],
                        help='Normalization type')
    parser.add_argument('--pos_type', type=str, default='learnable', choices=['sinusoidal', 'learnable'],
                        help='Position embedding type')

    # --- Training Hyperparameters ---
    parser.add_argument('--resume', type=str,
                        default=r'/mnt/afs/250010063/AP0004_Midterm/results/Transformer_large/best_model.pth',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=240, help='Batch size')
    parser.add_argument('--lr', type=float, default=8e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ==========================================
# 2. Masking Helper Functions
# ==========================================
def generate_square_subsequent_mask(sz, device):
    """Generate a look-ahead mask to prevent attending to future tokens."""
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, trg, device):
    """Create padding masks for encoder/decoder and look-ahead mask for decoder."""
    src_seq_len = src.shape[1]
    trg_seq_len = trg.shape[1]

    trg_mask = generate_square_subsequent_mask(trg_seq_len, device)
    src_padding_mask = (src == PAD_IDX)
    trg_padding_mask = (trg == PAD_IDX)
    
    return src_padding_mask, trg_padding_mask, trg_mask


# ==========================================
# 3. Training & Evaluation Functions
# ==========================================
def train_epoch(epoch, model, iterator, optimizer, criterion, clip, device, logger):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        
        # Shift target sequence
        trg_input = trg[:, :-1] # Input to decoder
        trg_label = trg[:, 1:]  # Ground truth labels
        
        src_padding_mask, trg_padding_mask, trg_mask = create_mask(src, trg_input, device)
        
        optimizer.zero_grad()
        
        output = model(
            src, 
            trg_input, 
            trg_mask=trg_mask,                   # Prevent peeking ahead
            src_key_padding_mask=src_padding_mask, # Mask encoder padding
            trg_key_padding_mask=trg_padding_mask  # Mask decoder padding
        )
        
        # Flatten for loss calculation
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg_label = trg_label.contiguous().view(-1)
        
        loss = criterion(output, trg_label)
        loss.backward()
        
        # Gradient clipping (optional/commented out in source)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        logger.log(f"Epoch{epoch}, Iter [{i+1}/{len(iterator)}] | Loss: {loss.item():.4f}")

    return epoch_loss / len(iterator)


def evaluate(model, dataloader, criterion, device, trg_vocab):
    """
    Evaluate on validation/test set.
    Returns: Average Loss (PPL related) and BLEU Score.
    """
    model.eval()
    epoch_loss = 0
    preds = []
    refs = []
    
    SOS_IDX = 1
    EOS_IDX = 2
    PAD_IDX = 0 
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(dataloader):
            src, trg = src.to(device), trg.to(device)
            
            # --- Part 1: Loss Calculation ---
            trg_input = trg[:, :-1]
            trg_label = trg[:, 1:]
            
            src_padding_mask, trg_padding_mask, trg_mask = create_mask(src, trg_input, device)
            
            output = model(
                src, 
                trg_input, 
                trg_mask=trg_mask, 
                src_key_padding_mask=src_padding_mask, 
                trg_key_padding_mask=trg_padding_mask 
            )
            
            output_dim = output.shape[-1]
            output_flat = output.contiguous().view(-1, output_dim)
            trg_label_flat = trg_label.contiguous().view(-1)
            
            loss = criterion(output_flat, trg_label_flat)
            epoch_loss += loss.item()
            
            # --- Part 2: BLEU Calculation (Greedy Decoding) ---
            src_mask = (src == PAD_IDX).to(device)
            max_len = trg.shape[1] + 10
            
            decoded_batch = greedy_decode(
                model, src, src_mask, max_len, SOS_IDX, EOS_IDX, device
            )
            
            # Post-processing
            batch_pred_ids = decoded_batch.tolist()
            batch_trg_ids = trg.tolist()
            
            for pred_ids, trg_ids in zip(batch_pred_ids, batch_trg_ids):
                # Clean Prediction
                clean_pred_ids = []
                for idx in pred_ids:
                    if idx == EOS_IDX: break
                    if idx not in (SOS_IDX, PAD_IDX):
                        clean_pred_ids.append(idx)
                
                # Clean Reference
                clean_trg_ids = []
                for idx in trg_ids:
                    if idx == EOS_IDX: break
                    if idx not in (SOS_IDX, PAD_IDX):
                        clean_trg_ids.append(idx)
                
                # Convert to string
                pred_tokens = [trg_vocab.idx2word.get(idx, '<unk>') for idx in clean_pred_ids]
                trg_tokens = [trg_vocab.idx2word.get(idx, '<unk>') for idx in clean_trg_ids]
                
                pred_str = " ".join(pred_tokens)
                ref_str = " ".join(trg_tokens)
                
                # Detokenization
                rep_rules = [
                    (" .", "."), (" ,", ","), (" ?", "?"), (" !", "!"), 
                    (" '", "'"), (" n't", "n't"), (" 's", "'s"), 
                    (" 'm", "'m"), (" 're", "'re"), (" 've", "'ve"),
                    (" 'll", "'ll"), (" 'd", "'d")
                ]
                for old, new in rep_rules:
                    pred_str = pred_str.replace(old, new)
                    ref_str = ref_str.replace(old, new)
                
                preds.append(pred_str)
                refs.append(ref_str)

    avg_loss = epoch_loss / len(dataloader)
    bleu_score = sacrebleu.corpus_bleu(preds, [refs]).score
    
    return avg_loss, bleu_score


# ==========================================
# 4. Main Function
# ==========================================
def main():
    # 1. Setup
    args = get_args()
    os.makedirs(os.path.join(args.save_path, args.exp), exist_ok=True)
    logger = TxtLogger(save_dir=os.path.join(args.save_path, args.exp))
    logger.log(f"Configurations: {args}")

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")
    
    # 2. Vocabulary & Data Loading
    raw_src_train, raw_trg_train = load_data_from_json(os.path.join(args.data_root, 'train_100k.jsonl'))
    vocap_pth = os.path.join(args.data_root, 'v1', 'vocab.pth')
    
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

    # 3. Datasets & DataLoaders
    train_cache = os.path.join(args.data_root, 'v1', 'train_100k_cache.pt')
    val_cache = os.path.join(args.data_root, 'v1', 'val.pt')
    test_cache = os.path.join(args.data_root, 'v1', 'test.pt')

    if os.path.exists(train_cache):
        train_dataset = NMTDataset(None, None, None, src_vocab, trg_vocab, max_len=160, cache_name=train_cache)
    else:
        train_dataset = NMTDataset(None, raw_src_train, raw_trg_train, src_vocab, trg_vocab, max_len=160, cache_name=train_cache)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = NMTDataset(os.path.join(args.data_root, 'valid.jsonl'), None, None, src_vocab, trg_vocab, max_len=1000, cache_name=val_cache)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    test_dataset = NMTDataset(os.path.join(args.data_root, 'test.jsonl'), None, None, src_vocab, trg_vocab, max_len=1000, cache_name=test_cache)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Initialize Model
    model = TransformerNMT(
        src_vocab_size=train_dataset.src_vocab.n_words,
        trg_vocab_size=train_dataset.trg_vocab.n_words,
        d_model=args.d_model,
        nhead=args.n_head,
        num_layers=args.n_layers,
        pos_embed_type=args.pos_type,
        norm_type=args.norm_type
    ).to(device)

    # Initialize weights
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    logger.log(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    # 5. Optimizer, Scheduler, Loss
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.98), 
        eps=1e-9, 
        weight_decay=1e-4
    )
    scheduler = CosineLRScheduler(
        optimizer, t_initial=args.epochs, warmup_t=10, warmup_lr_init=1e-5, cycle_limit=1
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    # 6. Training Loop
    best_blue = 0
    current_epoch = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_PPL': [], 'val_blue': [], 'lr': []}

    # Resume from checkpoint
    if args.resume != '':
        weights = torch.load(args.resume, map_location=device)
        model.load_state_dict(weights['model'])
        scheduler.load_state_dict(weights['scheduler'])
        optimizer.load_state_dict(weights['optimizer'])
        current_epoch = weights['epoch'] + 1
        best_blue = weights['performance']
        history = weights['history']
        
        logger.log(f"Resumed from epoch {current_epoch}. Evaluating on test set...")
        test_loss, test_blue = evaluate(model, test_loader, criterion, device, train_dataset.trg_vocab)
        logger.log(f"Test Blue Score on Resume: {test_blue:.2f}%")

    for epoch in range(current_epoch, args.epochs):
        train_loss = train_epoch(epoch, model, train_loader, optimizer, criterion, args.clip, device, logger)
        val_loss, val_blue = evaluate(model, val_loader, criterion, device, train_dataset.trg_vocab)
        
        scheduler.step(epoch)
        lr = optimizer.param_groups[0]['lr']

        # Record History
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_PPL'].append(math.exp(train_loss))
        history['val_blue'].append(val_blue)
        history['lr'].append(lr)
        
        logger.log(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val Loss: {val_loss:.3f} | Val Blue Score: {val_blue:.3f}')
        
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
            logger.log(f"Model saved to {args.save_path}")

    # Export History
    pd.DataFrame(history).to_csv(os.path.join(args.save_path, args.exp, 'training_history.csv'), index=False)
    logger.log(f"Training history saved")

    # Final Test
    weights = torch.load(os.path.join(args.save_path, args.exp, 'best_model.pth'), map_location=device)
    model.load_state_dict(weights['model'])
    test_loss, test_blue = evaluate(model, test_loader, criterion, device, train_dataset.trg_vocab)
    logger.log(f"Final Test Blue Score: {test_blue:.2f}%")


if __name__ == '__main__':
    main()