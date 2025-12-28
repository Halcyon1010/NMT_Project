import os
import torch
import sacrebleu
import math
import collections
from bert_score import score as bert_score_func
from collections import Counter
from typing import List, Sequence, Union
import re

class TxtLogger:
    def __init__(self, save_dir, filename="train_log.txt"):
        self.log_path = os.path.join(save_dir, filename)
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write("=== Training Log Started ===\n")
        else:
            with open(self.log_path, 'a') as f:
                f.write("=== Training Log Resume ===\n")

    def log(self, message):
        print(message)
        with open(self.log_path, 'a') as f:
            f.write(message + "\n")

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    src = src.to(device)
    if src_mask is not None:
        src_mask = src_mask.to(device)

    # 1. 动态获取 Batch Size
    batch_size = src.shape[0]

    # Encoder
    # 兼容不同的模型写法 (model.encode 或 model.encoder)
    memory = model.encode(src, src_mask) if hasattr(model, 'encode') else model.encoder(src)
    
    # Decoder Init
    # 修改：使用 src 的 batch_size，而不是写死 1
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type(torch.long).to(device)
    
    # 用于记录哪些句子已经生成结束符 (Batch推理时需要)
    finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

    for i in range(max_len - 1):
        # 确保 mask 在正确的 device 上
        sz = ys.shape[1]
        tgt_mask = torch.triu(torch.ones((sz, sz), device=device)) == 1
        tgt_mask = tgt_mask.transpose(0, 1).float()
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, 0.0)
        
        # Decode
        out = model.decode(ys, memory, tgt_mask)
        
        # Projection: 取最后一个时间步
        prob = model.fc_out(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        
        # 修改：next_word 现在是一个 Batch 的 tensor [batch_size]
        # 如果已经结束的句子，保持填充 end_symbol 或者 pad (这里简化处理，继续生成但不影响结果截断)
        
        # 拼接
        next_word = next_word.unsqueeze(1) # [batch_size, 1]
        ys = torch.cat([ys, next_word], dim=1)
        
        # 如果是单条推理（Batch=1），遇到 end_symbol 直接退出
        # 如果是 Batch 推理，通常不在这里 break，而是生成到固定长度后由后处理截断
        if batch_size == 1 and next_word.item() == end_symbol:
            break
            
    return ys

import torch

def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device, beam_width=3):
    """
    简易版 Beam Search (Batch Size = 1)
    """
    model.eval()
    with torch.no_grad():
        # Encoder 只需跑一次
        memory = model.encode(src, src_mask)
        
        # beams 列表: 存储 (累加得分, 生成序列张量)
        # 初始状态:得分为0, 序列只有 [SOS]
        start_seq = torch.LongTensor([[start_symbol]]).to(device)
        beams = [(0.0, start_seq)]
        
        finished_beams = []
        
        for _ in range(max_len):
            new_beams = []
            
            for score, seq in beams:
                # 1. 准备解码器的输入 (Mask)
                tgt_mask = (torch.triu(torch.ones((seq.shape[1], seq.shape[1]), device=device)) == 1).transpose(0, 1)
                tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, 0.0)
                
                # 2. Decode
                out = model.decode(seq, memory, tgt_mask)
                # 取最后一个时间步
                prob = model.fc_out(out[:, -1]) 
                # 取 Log Softmax (方便累加得分)
                log_prob = torch.log_softmax(prob, dim=1)
                
                # 3. 取前 beam_width 个最佳候选
                topk_probs, topk_ids = torch.topk(log_prob, beam_width, dim=1)
                
                for i in range(beam_width):
                    token_id = topk_ids[0, i].item()
                    token_prob = topk_probs[0, i].item()
                    
                    # 累加分数
                    new_score = score + token_prob
                    
                    # 遇到 EOS，说明这句话生成完了
                    if token_id == end_symbol:
                        finished_beams.append((new_score, seq))
                    else:
                        # 没结束，拼接新 Token，加入候选
                        new_seq = torch.cat([seq, torch.LongTensor([[token_id]]).to(device)], dim=1)
                        new_beams.append((new_score, new_seq))
            
            # 如果所有 beam 都结束了，退出
            if not new_beams:
                break
                
            # 4. 剪枝：保留得分最高的 beam_width 个
            # 为了防止偏向短句，通常可以用长度归一化 (score / length^alpha)，这里暂时用纯分数
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
            
            # 如果已经收集够了完成的句子，也可以提前退出 (可选)
            if len(finished_beams) >= beam_width:
                break
        
        # 最终选择：从 finished_beams 和还在跑的 beams 里选最好的
        all_candidates = finished_beams + beams
        # 简单长度惩罚：除以长度，让长句子更有竞争力
        best_beam = max(all_candidates, key=lambda x: x[0] / (x[1].shape[1]**0.6))
        
        return best_beam[1]
    

def batch_beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device, beam_width=3):
    """
    支持 Batch 并行的 Beam Search
    src: [batch_size, src_len]
    """
    batch_size = src.shape[0]
    
    # 1. Encoder (只跑一次)
    # memory: [batch_size, src_len, d_model]
    memory = model.encode(src, src_mask) if hasattr(model, 'encode') else model.encoder(src)
    
    # --- 核心操作：将 Memory 扩展 beam_width 倍 ---
    # 变成了 [batch_size * beam_width, src_len, d_model]
    # 这样 Decoder 可以一次性处理所有 Beam 的候选
    memory = memory.repeat_interleave(beam_width, dim=0)
    
    # src_mask 也需要扩展
    if src_mask is not None:
        src_mask = src_mask.repeat_interleave(beam_width, dim=0)

    # 2. 初始化 Decoder 输入
    # [batch_size * beam_width, 1] 全是 SOS
    ys = torch.ones(batch_size * beam_width, 1).fill_(start_symbol).type(torch.long).to(device)
    
    # 3. 初始化分数
    # scores: [batch_size, beam_width]
    # 第一个 beam 分数为 0，其他设为 -inf (强制模型初始只能选第一条路，否则 3 条路都是一样的)
    scores = torch.full((batch_size, beam_width), float("-inf"), device=device)
    scores[:, 0] = 0.0
    
    # 把它拉平方便后续计算: [batch_size * beam_width]
    scores = scores.view(-1) 
    
    # 记录哪些句子已经生成了 EOS
    # [batch_size * beam_width]
    finished_mask = torch.zeros(batch_size * beam_width, dtype=torch.bool, device=device)

    for i in range(max_len):
        # 4. 构建 Mask
        tgt_mask = (torch.triu(torch.ones((ys.shape[1], ys.shape[1]), device=device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, 0.0)
        
        # 5. Decoder 前向传播
        # out: [batch * beam, seq_len, d_model]
        out = model.decode(ys, memory, tgt_mask)
        
        # 6. 取最后一个时间步的 Log Softmax
        # prob: [batch * beam, vocab_size]
        prob = model.fc_out(out[:, -1])
        log_prob = torch.log_softmax(prob, dim=1)
        
        # 7. 计算总分 = 历史分 + 当前分
        # scores 扩展维度: [batch*beam, 1] + [batch*beam, vocab] -> [batch*beam, vocab]
        # 对于已经结束的句子 (finished_mask=True)，不再累加概率，或者给 EOS 极大概率(这里简化处理，不做特殊屏蔽，依靠后续 TopK)
        next_scores = scores.unsqueeze(1) + log_prob
        
        # 8. 维度重塑，以便在每个 Batch 内部找 TopK
        # [batch, beam * vocab]
        next_scores = next_scores.view(batch_size, -1)
        
        # 9. TopK 选择
        # best_scores: [batch, beam]
        # best_indices: [batch, beam] (取值范围 0 ~ beam*vocab-1)
        best_scores, best_indices = torch.topk(next_scores, beam_width, dim=1)
        
        # 10. 解析 TopK 的索引
        # 算出它属于原来的哪个 beam (beam_idx) 以及是哪个 token (token_idx)
        vocab_size = log_prob.shape[1]
        beam_indices = best_indices.div(vocab_size, rounding_mode='floor') # 属于哪个旧 beam
        token_indices = best_indices % vocab_size                          # 是哪个词
        
        # 11. 重新构建 ys (生成序列)
        # 我们需要根据 beam_indices 重新排列 ys，然后把新 token 拼上去
        
        # 计算全局索引以便从 [batch * beam, seq_len] 中取数
        # base_indices: [0, beam, 2*beam, ...] 对应每个 batch 的起始位置
        base_indices = torch.arange(batch_size, device=device).unsqueeze(1) * beam_width
        # global_beam_indices: [batch, beam] -> flatten -> [batch * beam]
        global_beam_indices = (base_indices + beam_indices).view(-1)
        
        # 重排旧序列
        ys = ys[global_beam_indices]
        
        # 拼接新 token
        new_tokens = token_indices.view(-1, 1)
        ys = torch.cat([ys, new_tokens], dim=1)
        
        # 更新分数
        scores = best_scores.view(-1)
        
        # 检查是否遇到 EOS (这里简化处理，遇到 EOS 的序列如果不被挤下去就保留)
        # 实际严谨的 Beam Search 会把结束的移入 finished 列表，这里为了并行效率保持形状不变
        is_eos = (new_tokens.view(-1) == end_symbol)
        finished_mask = finished_mask | is_eos
        
        # 如果所有 beam 都遇到过 EOS，或者达到 max_len，通常这里不好做 Early Stop，因为是 Batch 并行
        # 可以判断是否整个 Batch 的所有 Beam 都结束了，但为了代码简单，通常跑满或大部分跑满
        if finished_mask.all():
            break
            
    # 12. 最终选择：取每个 Batch 中分数最高的那个 Beam
    # reshape scores: [batch, beam]
    final_scores = scores.view(batch_size, beam_width)
    best_beam_idxs = final_scores.argmax(dim=1) # [batch]
    
    # 提取对应的序列
    base_indices = torch.arange(batch_size, device=device) * beam_width
    best_global_idxs = base_indices + best_beam_idxs
    
    best_sequences = ys[best_global_idxs]
    
    return best_sequences

Tokens = List[str]
Sentence = Union[str, Tokens]
def _tokenize_ws(x: Sentence) -> Tokens:
    """Whitespace tokenization. Your preds/refs are already space-joined."""
    return x.split() if isinstance(x, str) else list(x)

def _ngrams(tokens: Tokens, n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def _closest_ref_len(cand_len: int, ref_lens: List[int]) -> int:
    # Closest reference length; tie -> shorter
    return min(ref_lens, key=lambda rl: (abs(rl - cand_len), rl))

def bleu4(
    candidates: Sequence[Sentence],
    references: Sequence[Sequence[Sentence]],  # 每条候选可对应多条参考；你现在是 1 条参考，所以传 [[ref], [ref], ...]
) -> float:
    """
    BLEU-4 exactly following your screenshot:

    precision_n = sum_clip_count / sum_count  (corpus-level)
    BLEU-4 = min(1, output_length / reference_length) * Π_{i=1..4} precision_i
    """
    if len(candidates) != len(references):
        raise ValueError("candidates and references must have the same length")

    clipped_totals = [0, 0, 0, 0, 0]  # index 1..4
    count_totals   = [0, 0, 0, 0, 0]

    output_length = 0
    reference_length = 0

    for cand, refs in zip(candidates, references):
        cand_tok = _tokenize_ws(cand)
        refs_tok = [_tokenize_ws(r) for r in refs]
        if len(refs_tok) == 0:
            raise ValueError("each candidate must have at least one reference")

        # lengths
        output_length += len(cand_tok)
        ref_lens = [len(r) for r in refs_tok]
        reference_length += _closest_ref_len(len(cand_tok), ref_lens)

        # n-grams
        for n in range(1, 5):
            cand_counts = _ngrams(cand_tok, n)
            total_cand = sum(cand_counts.values())
            count_totals[n] += total_cand
            if total_cand == 0:
                continue

            # max ref count per n-gram over refs
            max_ref_counts = Counter()
            for r in refs_tok:
                rc = _ngrams(r, n)
                for ng, ct in rc.items():
                    if ct > max_ref_counts[ng]:
                        max_ref_counts[ng] = ct

            clipped = 0
            for ng, ct in cand_counts.items():
                clipped += min(ct, max_ref_counts.get(ng, 0))
            clipped_totals[n] += clipped

    precisions = []
    for n in range(1, 5):
        if count_totals[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_totals[n] / count_totals[n])

    bp = 0.0 if reference_length == 0 else min(1.0, output_length / reference_length)

    bleu = bp
    for p in precisions:
        bleu *= p
    return float(bleu)

class MetricCalculator:
    def __init__(self, device="cuda"):
        self.device = device

    def compute_ppl(self, avg_loss):
        try:
            return math.exp(avg_loss)
        except OverflowError:
            return float("inf")

    def compute_bleu(self, preds, refs):
        return sacrebleu.corpus_bleu(preds, [refs]).score

    def compute_bertscore(self, preds, refs):
        P, R, F1 = bert_score_func(preds, refs, lang="en", device=self.device, verbose=False)
        return F1.mean().item() * 100
    def compute_bleu_4(self, preds, refs):
        
        
        
        def simple_tokenize(text):
            
            return re.findall(r'\w+|[^\w\s]', text)

        
        cand_tokens_list = [simple_tokenize(p) for p in preds]
        ref_tokens_list = [simple_tokenize(r) for r in refs]

        
        
        
        precisions = []
        for n in range(1, 5):
            numerator = 0
            denominator = 0
            
            for cand_tokens, ref_tokens in zip(cand_tokens_list, ref_tokens_list):
                cand_ngrams = self._get_ngrams(cand_tokens, n)
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                
                cand_counts = collections.Counter(cand_ngrams)
                ref_counts = collections.Counter(ref_ngrams)
                
                denominator += sum(cand_counts.values())
                
                for ngram, count in cand_counts.items():
                    numerator += min(count, ref_counts.get(ngram, 0))
            
            if denominator == 0:
                precisions.append(0.0)
            else:
                precisions.append(numerator / denominator)

        
        output_length = sum(len(c) for c in cand_tokens_list)
        reference_length = sum(len(r) for r in ref_tokens_list)
        
        if reference_length == 0:
            bp = 0.0
        else:
            bp = min(1.0, output_length / reference_length)

        
        score = bp
        for p in precisions:
            score *= p
            
        return score * 100
    def _get_ngrams(self, tokens, n):
        
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

@torch.no_grad()
def beam_decode_rnn(
    model,
    src,                     # [src_len, B]
    beam_width=5,
    max_len=200,
    pad_id=0,
    sos_id=1,
    eos_id=2,
    device=None,
    length_penalty_alpha=0.0 # 0.0 = no penalty
):
    """
    Performs beam search decoding (sample-by-sample for stability).
    Returns: pred_ids [B, T]
    """
    model.eval()
    if device is None:
        device = src.device

    src = src.to(device)
    src_len, B = src.shape

    # Encode batch
    encoder_outputs, enc_hidden = model.encoder(src)  # [src_len, B, H]
    src_mask = (src.transpose(0, 1) == pad_id)        # [B, src_len]

    all_best = []
    
    # Process each sample in the batch
    for b in range(B):
        # Extract single sample data
        enc_out_b = encoder_outputs[:, b:b+1, :]          # [src_len, 1, H]
        
        if isinstance(enc_hidden, torch.Tensor):
            hid_b = enc_hidden[:, b:b+1, :].contiguous()  # [n_layers, 1, H]
        else:
            hid_b = enc_hidden  # Tuple for LSTM

        src_mask_b = src_mask[b:b+1, :]                   # [1, src_len]

        # Beam: list of (score, seq, hidden, ended_flag)
        beam = [(0.0, [sos_id], hid_b, False)]

        for _ in range(max_len):
            candidates = []
            for score, seq, hid, ended in beam:
                if ended or seq[-1] == eos_id:
                    candidates.append((score, seq, hid, True))
                    continue

                dec_input = torch.tensor([[seq[-1]]], device=device)  # [1, 1]
                logp, new_hid, _ = model.decoder(
                    dec_input, hid, enc_out_b, src_mask=src_mask_b
                )  # logp: [1, vocab]

                # Top-k candidates
                topv, topi = torch.topk(logp, k=beam_width, dim=1)
                for k in range(beam_width):
                    wid = int(topi[0, k].item())
                    wlogp = float(topv[0, k].item())

                    new_seq = seq + [wid]
                    new_score = score + wlogp
                    candidates.append((new_score, new_seq, new_hid, wid == eos_id))

            # Select top candidates with Length Penalty
            def norm_score(item):
                s, seq, _, _ = item
                if length_penalty_alpha <= 0:
                    return s
                # GNMT length penalty
                lp = ((5 + len(seq)) / 6) ** length_penalty_alpha
                return s / lp

            candidates = sorted(candidates, key=norm_score, reverse=True)[:beam_width]
            beam = candidates

            if all(ended for _, _, _, ended in beam):
                break

        best_seq = beam[0][1]
        all_best.append(best_seq)

    # Pad sequences to create batch tensor
    maxT = max(len(x) for x in all_best)
    out = torch.full((B, maxT), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(all_best):
        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

    return out


# ==========================================
# 3) Decoding Functions
# ==========================================
@torch.no_grad()
def greedy_decode_rnn(
    model,
    src,                     # [src_len, B]
    max_len,
    pad_id,
    sos_id,
    eos_id,
    device,
    src_mask=None,           # [B, src_len]
):
    """
    Performs greedy decoding.
    Returns: pred_ids [B, T]
    """
    model.eval()
    src = src.to(device)

    # Encode
    encoder_outputs, hidden = model.encoder(src)  # encoder_outputs: [src_len, B, H]

    B = src.shape[1]
    input_t = torch.full((B,), sos_id, dtype=torch.long, device=device)  # [B]

    preds = [input_t]
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        # Decode step
        out, hidden, _ = model.decoder(input_t.unsqueeze(0), hidden, encoder_outputs)  # out: [B, vocab]

        next_t = out.argmax(dim=1)  # [B]
        
        # If finished, keep outputting EOS (or pad)
        next_t = torch.where(finished, torch.full_like(next_t, eos_id), next_t)

        preds.append(next_t)
        finished = finished | (next_t == eos_id)

        input_t = next_t
        if torch.all(finished):
            break

    pred_ids = torch.stack(preds, dim=1)  # [B, T]
    return pred_ids
