import random, torch, torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, cfg, use_attention):
        super().__init__()
        self.use_attention, self.cfg = use_attention, dict(cfg)
        self.encoder = Encoder(src_vocab_size, cfg["emb_dim"], cfg["enc_hidden"], cfg["dec_hidden"], cfg["num_layers"],
                               cfg["bidirectional"], cfg["dropout"])
        self.decoder = Decoder(tgt_vocab_size, cfg["emb_dim"], cfg["dec_hidden"], self.encoder.enc_dim, cfg["num_layers"],
                               cfg["dropout"], use_attention, cfg["attn_dim"])
        self.sos_id, self.eos_id, self.pad_id = 2, 3, 0

    def encode(self, src, src_len):
        enc_out, state = self.encoder(src, src_len)
        mask = src != self.pad_id                                    # (B,S) True = real token
        return enc_out, state, mask

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=1.0):
        """tgt = <SOS> y1 ... yn <EOS> (B,T). Returns logits (B, T-1, V) predicting tgt[:, 1:].
        Step t consumes tgt[:, t-1] (teacher forcing) or the model's own previous argmax."""
        enc_out, state, mask = self.encode(src, src_len)
        T = tgt.shape[1]
        inp, outs = tgt[:, 0], []
        for t in range(1, T):
            logits, state, _ = self.decoder.step(inp, state, enc_out, mask)
            outs.append(logits)
            use_gold = random.random() < teacher_forcing_ratio
            inp = tgt[:, t] if use_gold else logits.argmax(-1)
        return torch.stack(outs, dim=1)

    @torch.no_grad()
    def generate(self, src, src_len, max_len=40, return_attention=False):
        """Greedy decoding. Never sees the reference. Returns ids (B, <=max_len) [+ attention (B,T,S)]."""
        self.eval()
        enc_out, state, mask = self.encode(src, src_len)
        B = src.size(0)
        inp = torch.full((B,), self.sos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        ids, atts = [], []
        for _ in range(max_len):
            logits, state, w = self.decoder.step(inp, state, enc_out, mask)
            inp = logits.argmax(-1).masked_fill(done, self.pad_id)   # finished rows stay PAD
            ids.append(inp)
            done = done | (inp == self.eos_id)
            if w is not None:
                atts.append(w)
            if done.all():
                break
        ids = torch.stack(ids, 1)
        if return_attention:
            return ids, (torch.stack(atts, 1) if atts else None)
        return ids

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
