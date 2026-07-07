"""
network.py -- Assembles the full model:

    tokens --Embedding--> +PositionalEncoding --SelfAttention--> masked mean-pool
           --Dense+ReLU--> --Dense--> logits --SoftmaxCrossEntropy--> loss

Every arrow above is implemented with a hand-derived backward() in
core.py / attention.py. This file wires them together and chains the
gradients end-to-end -- this IS backpropagation through a full network,
done manually rather than via autograd.
"""

import numpy as np
from core import Dense, ReLU, SoftmaxCrossEntropy
from attention import Embedding, SelfAttention, positional_encoding


class TransformerIntentClassifier:
    def __init__(self, vocab_size, max_len, d_model=32, d_k=32, hidden=32,
                 num_classes=2, seed=42):
        # BUG FIXED: every layer previously received the exact same `seed`
        # value, which meant their weight matrices were initialized from
        # the same random draw sequence -- e.g. dense1.W and dense2.W
        # shared identical values in their overlapping leading entries
        # (verified: both started with [0.088, -0.300, ...] before this
        # fix). Independent, decorrelated initialization is standard
        # practice precisely to avoid this kind of accidental symmetry
        # between layers; each component now gets a distinct derived seed.
        self.embedding = Embedding(vocab_size, d_model, seed=seed)
        self.pos_enc = positional_encoding(max_len, d_model)  # fixed, not learned
        self.attn = SelfAttention(d_model, d_k, seed=seed + 1)
        self.dense1 = Dense(d_k, hidden, seed=seed + 2)
        self.relu = ReLU()
        self.dense2 = Dense(hidden, num_classes, seed=seed + 3)
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, token_ids, key_mask):
        """
        token_ids: [B, N] int array of vocabulary indices (padded with 0)
        key_mask:  [B, N] 1.0 for real tokens, 0.0 for padding
        """
        emb = self.embedding.forward(token_ids)          # [B, N, d_model]
        emb = emb + self.pos_enc[: emb.shape[1]]          # inject position info
        attn_out = self.attn.forward(emb, key_mask)       # [B, N, d_k]

        mask3 = key_mask[:, :, None]                      # [B, N, 1]
        summed = (attn_out * mask3).sum(axis=1)            # [B, d_k]
        counts = np.clip(mask3.sum(axis=1), 1, None)        # avoid /0
        pooled = summed / counts                            # masked mean-pool

        self._pool_cache = (mask3, counts)

        h = self.dense1.forward(pooled)
        h = self.relu.forward(h)
        logits = self.dense2.forward(h)
        return logits

    def compute_loss(self, logits, y_idx):
        return self.loss_fn.forward(logits, y_idx)

    def backward(self):
        dlogits = self.loss_fn.backward()
        dh = self.dense2.backward(dlogits)
        dh = self.relu.backward(dh)
        dpooled = self.dense1.backward(dh)

        mask3, counts = self._pool_cache
        # backward of masked mean-pool: distribute gradient equally across
        # real tokens, zero out padding positions.
        # dpooled: [B, d_k], counts: [B, 1] -> reshape to [B, 1, 1] to broadcast
        # against mask3: [B, N, 1] and produce dattn_out: [B, N, d_k]
        dattn_out = (dpooled[:, None, :] / counts[:, :, None]) * mask3

        dembed = self.attn.backward(dattn_out)
        self.embedding.backward(dembed)  # positional encoding is fixed, no grad needed there

    def step(self, lr):
        self.embedding.step(lr)
        self.attn.step(lr)
        self.dense1.step(lr)
        self.dense2.step(lr)

    def all_params_flat(self):
        """Returns (name, array, grad) tuples for gradient checking."""
        out = [("embedding.table", self.embedding.table, self.embedding.dtable)]
        out += [(f"attn.{n}", getattr(self.attn, n), getattr(self.attn, "d" + n))
                for n in ("Wq", "Wk", "Wv")]
        out += [(f"dense1.{n}", v, g) for n, v, g in self.dense1.params()]
        out += [(f"dense2.{n}", v, g) for n, v, g in self.dense2.params()]
        return out


class OrderBlindClassifier:
    """
    A deliberately ORDER-BLIND baseline for the ablation study in
    ablation_order_sensitivity.py: same Embedding + Dense head as
    TransformerIntentClassifier, but with NO positional encoding and NO
    attention -- just a masked mean-pool straight over token embeddings.

    Mean-pooling is a permutation-invariant operation (sum/count doesn't
    care about order), and without positional encoding added beforehand,
    there is no other source of order information anywhere in the
    architecture. This makes it a true bag-of-embeddings model: provably
    incapable of using word order, by construction rather than by
    empirical observation. That's the whole point of using it as a
    control in the ablation.
    """

    def __init__(self, vocab_size, d_model=24, hidden=24, num_classes=2, seed=42):
        self.embedding = Embedding(vocab_size, d_model, seed=seed)
        self.dense1 = Dense(d_model, hidden, seed=seed + 1)
        self.relu = ReLU()
        self.dense2 = Dense(hidden, num_classes, seed=seed + 2)
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, token_ids, key_mask):
        emb = self.embedding.forward(token_ids)  # [B, N, d_model] -- no + pos_enc here
        mask3 = key_mask[:, :, None]
        summed = (emb * mask3).sum(axis=1)
        counts = np.clip(mask3.sum(axis=1), 1, None)
        pooled = summed / counts
        self._pool_cache = (mask3, counts)

        h = self.dense1.forward(pooled)
        h = self.relu.forward(h)
        logits = self.dense2.forward(h)
        return logits

    def compute_loss(self, logits, y_idx):
        return self.loss_fn.forward(logits, y_idx)

    def backward(self):
        dlogits = self.loss_fn.backward()
        dh = self.dense2.backward(dlogits)
        dh = self.relu.backward(dh)
        dpooled = self.dense1.backward(dh)

        mask3, counts = self._pool_cache
        demb = (dpooled[:, None, :] / counts[:, :, None]) * mask3
        self.embedding.backward(demb)

    def step(self, lr):
        self.embedding.step(lr)
        self.dense1.step(lr)
        self.dense2.step(lr)

    def all_params_flat(self):
        out = [("embedding.table", self.embedding.table, self.embedding.dtable)]
        out += [(f"dense1.{n}", v, g) for n, v, g in self.dense1.params()]
        out += [(f"dense2.{n}", v, g) for n, v, g in self.dense2.params()]
        return out
