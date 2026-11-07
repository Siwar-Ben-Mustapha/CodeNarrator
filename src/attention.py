import torch, torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Classical additive (Bahdanau) encoder-decoder attention -- NOT Transformer self-attention.

        score(s, h_j) = v^T tanh(W_s s + W_h h_j)      one score per encoder position j
        alpha         = softmax(score)  (PAD positions set to -inf before the softmax)
        context       = sum_j alpha_j h_j
    """

    def __init__(self, enc_dim, dec_hidden, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_hidden, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_state, enc_out, mask):
        """dec_state (B, H_dec); enc_out (B, S, enc_dim); mask (B, S) True = real token.
        returns context (B, enc_dim), weights (B, S)"""
        scores = self.v(torch.tanh(self.W_h(enc_out) + self.W_s(dec_state).unsqueeze(1))).squeeze(-1)  # (B,S)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), enc_out).squeeze(1)
        return context, weights
