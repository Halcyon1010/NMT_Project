import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ==========================================
# 1. Constants
# ==========================================
PAD_TOKEN = 0  # Padding
SOS_TOKEN = 1  # Start of Sequence
EOS_TOKEN = 2  # End of Sequence
UNK_TOKEN = 3  # Unknown Token


# ==========================================
# 2. Model Components
# ==========================================
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_TOKEN)
        # Config: Multi-layer unidirectional GRU
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, 
                          bidirectional=False, dropout=dropout)

    def forward(self, input_seq, hidden=None):
        # input_seq: [seq_len, batch_size]
        embedded = self.embedding(input_seq)
        outputs, hidden = self.gru(embedded, hidden)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method  # 'dot', 'general', or 'concat'
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
            nn.init.normal_(self.v, mean=0.0, std=0.1)

    def forward(self, hidden, encoder_outputs, src_mask=None):
        # hidden: [1, batch_size, hidden_size] (decoder current hidden state)
        # encoder_outputs: [max_length, batch_size, hidden_size]
        
        # Calculate alignment scores
        if self.method == 'dot':
            # Score = H_dec * H_enc
            attn_energies = torch.sum(hidden * encoder_outputs, dim=2)
            
        elif self.method == 'general':
            # Score = H_dec * W * H_enc
            energy = self.attn(encoder_outputs)
            attn_energies = torch.sum(hidden * energy, dim=2)
            
        elif self.method == 'concat':
            # Score = v^T * tanh(W * [H_dec; H_enc])
            hidden_expanded = hidden.expand(encoder_outputs.size(0), -1, -1)
            combined = torch.cat((hidden_expanded, encoder_outputs), 2)
            energy = torch.tanh(self.attn(combined))
            attn_energies = torch.sum(self.v * energy, dim=2)
        
        # Transpose to [batch_size, seq_len] for masking and softmax
        attn_energies = attn_energies.t()
        
        if src_mask is not None:
            # src_mask: True for PAD positions, set to -inf to ignore
            attn_energies = attn_energies.masked_fill(src_mask, float('-inf'))
        
        # Return attention weights: [batch_size, 1, seq_len]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_method, hidden_size, output_size, n_layers=2, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_method = attn_method
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_TOKEN)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, 
                          bidirectional=False, dropout=dropout)
        
        self.attn = Attention(attn_method, hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, encoder_outputs, src_mask=None):
        # input_step: [1, batch_size] (one word at a time)
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # 1. Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # 2. Calculate attention weights from current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs, src_mask=src_mask)
        
        # 3. Calculate context vector (weighted sum of encoder outputs)
        # context: [batch_size, 1, hidden_size]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # 4. Concatenate context vector and GRU output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # 5. Predict next token
        output = self.out(concat_output)
        output = F.log_softmax(output, dim=1)

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size
        
        # Create mask for source padding
        src_mask = (src.transpose(0, 1) == PAD_TOKEN)
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Encoder forward
        encoder_outputs, hidden = self.encoder(src)

        # Initialize decoder input with SOS token
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input.unsqueeze(0), hidden, encoder_outputs, src_mask=src_mask)
            outputs[t] = output
            
            # Teacher Forcing: decides whether to feed the previous ground truth or prediction
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            
            input = trg[t] if use_teacher_forcing else top1

        return outputs
