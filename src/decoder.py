import torch, torch.nn as nn
from attention import BahdanauAttention


class Decoder(nn.Module):
    """One-step LSTM decoder. use_attention=False -> baseline (only the bridged encoder state).

    Baseline:   input_t = emb(y_{t-1})                      logits_t = W [o_t]
    Attention:  c_t = Attn(s_{t-1}, enc_out)                (s_{t-1} = top-layer hidden state)
                input_t = [emb(y_{t-1}); c_t]               logits_t = W [o_t ; c_t]
    """

    def __init__(self, vocab_size, emb_dim, hidden, enc_dim, num_layers=1, dropout=0.3, use_attention=False,
                 attn_dim=128, pad_id=0):
        super().__init__()
        self.use_attention, self.num_layers = use_attention, num_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(enc_dim, hidden, attn_dim) if use_attention else None
        self.lstm = nn.LSTM(emb_dim + (enc_dim if use_attention else 0), hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden + (enc_dim if use_attention else 0), vocab_size)

    def step(self, token, state, enc_out=None, mask=None):
        """token (B,), state=(h,c) each (L,B,H). returns logits (B,V), new state, attn weights (B,S) or None"""
        x = self.drop(self.embedding(token))                       # (B, E)
        weights, ctx = None, None
        if self.use_attention:
            ctx, weights = self.attention(state[0][-1], enc_out, mask)
            x = torch.cat([x, ctx], dim=-1)
        o, state = self.lstm(x.unsqueeze(1), state)
        o = o.squeeze(1)
        feat = torch.cat([o, ctx], dim=-1) if self.use_attention else o
        return self.out(self.drop(feat)), state, weights
