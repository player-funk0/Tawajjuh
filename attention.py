"""
attention.py -- Scaled dot-product self-attention, derived and implemented
from raw linear algebra. This is the core mechanism behind Transformers
(and models like Claude), built here with hand-derived gradients.

===========================================================================
THE MATH
===========================================================================

Given a sequence of token embeddings X in R^(N x d_model) (N tokens, each
a d_model-dim vector), self-attention lets each token "look at" every
other token and aggregate information, weighted by relevance.

1. Project X into Query, Key, Value spaces:
       Q = X Wq,   K = X Wk,   V = X Wv        (each in R^(N x d_k))

2. Compute similarity scores between every pair of tokens via dot product:
       S_ij = (Q_i . K_j) / sqrt(d_k)

   The 1/sqrt(d_k) scaling exists because for random Q, K with unit
   variance components, Var(Q_i . K_j) = d_k. Without scaling, dot
   products grow with dimension and push softmax into saturated regions
   with vanishing gradients. Dividing by sqrt(d_k) keeps variance ~1
   regardless of dimension.

3. Normalize each row into a probability distribution over "which tokens
   to attend to":
       A_i = softmax(S_i)      (row-wise softmax, A_ij = attention weight
                                 token i places on token j)

4. Aggregate values weighted by attention:
       O = A V                 (O in R^(N x d_k))

===========================================================================
THE BACKWARD PASS (derived via chain rule)
===========================================================================

Given dL/dO from the next layer:

    dL/dV = A^T @ dL/dO
    dL/dA = dL/dO @ V^T

Backprop through row-wise softmax (standard softmax-Jacobian result,
using the same trick as in core.SoftmaxCrossEntropy but for a general
upstream gradient rather than cross-entropy specifically):

    dL/dS_i = A_i * (dL/dA_i - sum_j A_ij * dL/dA_ij)

Backprop through the scaled dot product S = QK^T / sqrt(d_k):

    dL/dQ = (dL/dS @ K) / sqrt(d_k)
    dL/dK = (dL/dS^T @ Q) / sqrt(d_k)

Backprop through the linear projections:

    dL/dWq = X^T @ dL/dQ,   dL/dX += dL/dQ @ Wq^T   (and similarly for Wk, Wv)

===========================================================================
WHY POSITIONAL ENCODING IS NEEDED
===========================================================================

Self-attention as defined above is PERMUTATION INVARIANT: shuffling the
order of tokens in X produces the same shuffled output, with no change in
the actual attention computation. This is a real theoretical property, and
it means raw attention has no notion of word order ("open notepad" and
"notepad open" would look identical). To fix this, we add a positional
encoding vector to each token embedding before attention, using the
sinusoidal scheme from Vaswani et al. (2017):

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Using sinusoids (rather than, say, learned positions) means the model can
generalize to positions not seen during training, and each dimension
oscillates at a different frequency, giving the network a unique,
smoothly-varying signature per position.
"""

import numpy as np
from core import softmax


def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / d_model)
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


class Embedding:
    """Learnable lookup table: token index -> d_model-dim vector."""

    def __init__(self, vocab_size, d_model, seed=None):
        rng = np.random.default_rng(seed)
        self.table = rng.normal(0, 0.1, size=(vocab_size, d_model))
        self._idx_cache = None
        self.dtable = None

    def forward(self, idx):  # idx: [B, N] int array
        self._idx_cache = idx
        return self.table[idx]  # [B, N, d_model]

    def backward(self, dY):  # dY: [B, N, d_model]
        self.dtable = np.zeros_like(self.table)
        np.add.at(self.dtable, self._idx_cache, dY)
        return None  # no further upstream (embedding is the input layer)

    def step(self, lr):
        self.table -= lr * self.dtable


class SelfAttention:
    """Single-head scaled dot-product self-attention (see module docstring)."""

    def __init__(self, d_model, d_k, seed=None):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)
        self.Wq = rng.normal(0, scale, (d_model, d_k))
        self.Wk = rng.normal(0, scale, (d_model, d_k))
        self.Wv = rng.normal(0, scale, (d_model, d_k))
        self.d_k = d_k
        self.dWq = self.dWk = self.dWv = None

    def forward(self, X, key_mask=None):
        """
        X: [B, N, d_model]
        key_mask: optional [B, N] boolean/0-1 array, True/1 = real token,
                  0 = padding. Padded keys are masked out by adding a large
                  negative number to their score before softmax, which
                  drives their attention weight to ~0 without changing the
                  backward math (gradients there are already ~0 since
                  A_ij ~ 0 for masked positions).
        """
        self.X = X
        self.Q = X @ self.Wq
        self.K = X @ self.Wk
        self.V = X @ self.Wv
        self.S = self.Q @ self.K.transpose(0, 2, 1) / np.sqrt(self.d_k)
        if key_mask is not None:
            additive_mask = (1.0 - key_mask.astype(np.float64))[:, None, :] * -1e9
            self.S = self.S + additive_mask
        self.A = softmax(self.S)  # row-wise: each query's distribution over keys
        self.O = self.A @ self.V
        return self.O

    def backward(self, dO):
        dA = dO @ self.V.transpose(0, 2, 1)
        dV = self.A.transpose(0, 2, 1) @ dO
        dS = self.A * (dA - np.sum(self.A * dA, axis=-1, keepdims=True))
        dQ = dS @ self.K / np.sqrt(self.d_k)
        dK = dS.transpose(0, 2, 1) @ self.Q / np.sqrt(self.d_k)

        self.dWq = np.einsum("bnd,bnk->dk", self.X, dQ)
        self.dWk = np.einsum("bnd,bnk->dk", self.X, dK)
        self.dWv = np.einsum("bnd,bnk->dk", self.X, dV)

        dX = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        return dX

    def step(self, lr):
        self.Wq -= lr * self.dWq
        self.Wk -= lr * self.dWk
        self.Wv -= lr * self.dWv

    def attention_weights(self):
        """Expose A for interpretability plots: which tokens the model attends to."""
        return self.A
