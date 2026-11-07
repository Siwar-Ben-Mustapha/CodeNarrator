import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    """Embedding -> (bi)LSTM. Identical for the baseline and the attention model.

    src:     (B, S) token ids          src_len: (B,) true lengths
    returns: outputs (B, S, enc_dim)   where enc_dim = hidden * (2 if bidirectional else 1)
             (h, c)  each (num_layers, B, dec_hidden): initial decoder states (via a learned bridge)
    """

    def __init__(self, vocab_size, emb_dim, hidden, dec_hidden, num_layers=1, bidirectional=True, dropout=0.3, pad_id=0):
        super().__init__()
        self.num_layers, self.hidden, self.bi = num_layers, hidden, bidirectional
        self.enc_dim = hidden * (2 if bidirectional else 1)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers, batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        # The final state of a bidirectional LSTM is [forward ; backward] (2*hidden). A linear
        # layer + tanh maps it to the decoder size so the decoder hidden size is free to choose.
        self.bridge_h = nn.Linear(self.enc_dim, dec_hidden)
        self.bridge_c = nn.Linear(self.enc_dim, dec_hidden)

    def _merge_directions(self, s):                      # (L*D, B, H) -> (L, B, D*H)
        L, D = self.num_layers, 2 if self.bi else 1
        s = s.view(L, D, s.size(1), self.hidden)
        return torch.cat([s[:, d] for d in range(D)], dim=-1)

    def forward(self, src, src_len):
        emb = self.drop(self.embedding(src))
        packed = pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm(packed)                  # packing => PAD never enters the final states
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        h = torch.tanh(self.bridge_h(self._merge_directions(h)))
        c = torch.tanh(self.bridge_c(self._merge_directions(c)))
        return out, (h, c)
