"""
network_torch.py -- PyTorch reimplementation of network.py.

Same architecture as the from-scratch numpy version (core.py / attention.py
/ network.py), but using PyTorch's autograd and nn.MultiheadAttention
instead of hand-derived backpropagation. This eliminates an entire class
of bugs (transposed matrices, missed gradient terms) by delegating the
math to a library that has been tested far more extensively than any
from-scratch implementation reasonably could be -- see report.md Section
13 for the full reasoning behind this choice and its honest limitations
in this specific build environment.

Bonus made trivial by using a real library: multi-head attention (4 heads
instead of 1) and the Adam optimizer, both flagged as "future work" in the
original report's Section 6/7, now included directly.

===========================================================================
IMPORTANT -- VERIFICATION STATUS
===========================================================================
This file could NOT be executed in the sandbox this project was built in
(no network access to install torch there). It has not been run or
gradient-checked by the assistant, unlike every other file in this
project. See report.md Section 13 and the sanity-check instructions in
train_torch.py for how to verify it yourself in ~30 seconds once you have
PyTorch installed.
"""

import torch
import torch.nn as nn


class TransformerIntentClassifierTorch(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=32, num_heads=4, hidden=32, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.register_buffer("pos_enc", self._positional_encoding(max_len, d_model))
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)

    @staticmethod
    def _positional_encoding(seq_len, d_model):
        # Identical sinusoidal scheme to attention.positional_encoding in
        # the numpy version -- kept the same so the two implementations
        # are conceptually comparable, not just accidentally similar.
        pos = torch.arange(seq_len).unsqueeze(1).float()
        i = torch.arange(d_model).unsqueeze(0).float()
        angle_rates = 1.0 / torch.pow(10000, (2 * (i // 2)) / d_model)
        angles = pos * angle_rates
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe

    def forward(self, token_ids, key_mask):
        """
        token_ids: LongTensor [B, N]
        key_mask:  FloatTensor [B, N], 1.0 = real token, 0.0 = padding
        """
        x = self.embedding(token_ids) + self.pos_enc[: token_ids.shape[1]]

        # nn.MultiheadAttention expects a boolean "True = ignore this
        # position" padding mask -- the inverse convention from our
        # key_mask, so we flip it here.
        attn_padding_mask = key_mask == 0  # [B, N] bool

        attn_out, attn_weights = self.attn(
            x, x, x, key_padding_mask=attn_padding_mask, need_weights=True, average_attn_weights=True
        )

        mask3 = key_mask.unsqueeze(-1)  # [B, N, 1]
        summed = (attn_out * mask3).sum(dim=1)
        counts = mask3.sum(dim=1).clamp(min=1)
        pooled = summed / counts

        h = self.relu(self.fc1(pooled))
        logits = self.fc2(h)
        return logits, attn_weights


class OrderBlindClassifierTorch(nn.Module):
    """Torch equivalent of network.OrderBlindClassifier -- no attention,
    no positional encoding, so it remains provably permutation-invariant."""

    def __init__(self, vocab_size, d_model=16, hidden=16, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.fc1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, token_ids, key_mask):
        x = self.embedding(token_ids)  # no positional encoding added
        mask3 = key_mask.unsqueeze(-1)
        summed = (x * mask3).sum(dim=1)
        counts = mask3.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        h = self.relu(self.fc1(pooled))
        return self.fc2(h)
