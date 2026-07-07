"""
core.py -- Neural network building blocks implemented from raw mathematics.

No autograd. No PyTorch. Every forward pass has a hand-derived backward pass
using the chain rule. Comments show the actual calculus.

Notation: for a layer y = f(x), we implement:
  - forward(x)  -> y
  - backward(dL/dy) -> dL/dx   (and accumulates dL/dW, dL/db internally)
"""

import numpy as np


class Dense:
    """
    Fully connected layer: y = xW + b

    Forward:  y = xW + b                         (x: [N, in], W: [in, out], y: [N, out])
    Backward (given dL/dy):
        dL/dW = x^T @ dL/dy                        (chain rule: dy/dW = x)
        dL/db = sum over batch of dL/dy             (dy/db = 1)
        dL/dx = dL/dy @ W^T                         (dy/dx = W)
    """

    def __init__(self, in_dim, out_dim, seed=None):
        rng = np.random.default_rng(seed)
        # He initialization: keeps variance of activations stable across layers.
        # var(W) = 2 / in_dim  is derived from wanting Var(xW) ~ Var(x) for ReLU nets.
        self.W = rng.normal(0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self._cache = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self._cache = x
        return x @ self.W + self.b

    def backward(self, dY):
        x = self._cache
        self.dW = x.T @ dY
        self.db = dY.sum(axis=0)
        dX = dY @ self.W.T
        return dX

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def params(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]


class ReLU:
    """
    f(x) = max(0, x)
    f'(x) = 1 if x > 0 else 0   (subgradient at 0 taken as 0)
    """

    def forward(self, x):
        self._mask = (x > 0).astype(x.dtype)
        return x * self._mask

    def backward(self, dY):
        return dY * self._mask


def softmax(z):
    """
    softmax(z)_i = exp(z_i) / sum_j exp(z_j)

    Subtracting max(z) before exponentiating is a numerical stability trick
    (log-sum-exp shift) -- it doesn't change the result since:
        exp(z_i - c) / sum(exp(z_j - c)) = exp(z_i) / sum(exp(z_j))
    but keeps exponents from overflowing.
    """
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class SoftmaxCrossEntropy:
    """
    Combines softmax with cross-entropy loss and gives a clean gradient.

    Cross-entropy: L = -sum_i y_i * log(p_i),   p = softmax(z)

    This is the negative log-likelihood of the true class under the model's
    predicted distribution. Equivalently, minimizing cross-entropy is
    minimizing the KL divergence D_KL(y || p) between the true one-hot
    distribution y and the predicted distribution p, since:
        D_KL(y||p) = sum y_i log(y_i/p_i) = -H(y) + CrossEntropy(y,p)
    and H(y) = 0 for a one-hot true distribution.

    Key derivation -- the gradient of the COMBINED softmax+cross-entropy
    with respect to the pre-softmax logits z simplifies beautifully to:
        dL/dz = p - y
    (This is why softmax and cross-entropy are almost always implemented
    together -- computing dL/dz directly avoids a much messier Jacobian
    of softmax alone.)
    """

    def forward(self, logits, y_true_idx):
        self.p = softmax(logits)
        self.y_idx = y_true_idx
        n = logits.shape[0]
        log_likelihood = -np.log(self.p[np.arange(n), y_true_idx] + 1e-12)
        return log_likelihood.mean()

    def backward(self):
        n = self.p.shape[0]
        dZ = self.p.copy()
        dZ[np.arange(n), self.y_idx] -= 1
        return dZ / n  # average over batch, matching the mean() in forward


def numerical_gradient(f, x, eps=1e-5):
    """
    Central-difference numerical gradient, used to VERIFY analytical
    (hand-derived) gradients are correct:

        df/dx_i ~= (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)

    This is the standard "gradient checking" technique used to catch bugs
    in manually-derived backpropagation -- a real engineering safeguard,
    not just a formality.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f()
        x[idx] = orig - eps
        f_minus = f()
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def clip_grad_norm(all_params_flat, max_norm):
    """
    Global gradient-norm clipping, implemented from scratch to match the
    rest of this project's "no autograd, but no black boxes either"
    approach.

    Computes the L2 norm of ALL gradients in the model treated as one
    flattened vector:
        total_norm = sqrt( sum_i sum(grad_i ** 2) )
    and if that exceeds max_norm, rescales every gradient array in place
    by max_norm / total_norm, so the combined gradient vector has norm
    exactly max_norm afterward while preserving its direction.

    This is standard practice in training deep/recurrent networks and
    directly targets a specific failure mode: an unusually large gradient
    on one bad batch/step causing a destructively large parameter update
    that the optimizer then has to recover from -- exactly the loss
    spikes observed with plain fixed-LR SGD on this project's dataset
    (see report.md Section 14).

    Mutates the gradient arrays in place (as returned by a model's
    all_params_flat()), matching how Dense.step() etc. read gradients
    directly off self.dW / self.db.
    """
    total_sq = 0.0
    grads = []
    for _, _, grad in all_params_flat:
        if grad is not None:
            total_sq += float(np.sum(grad ** 2))
            grads.append(grad)
    total_norm = np.sqrt(total_sq)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for grad in grads:
            grad *= scale

    return total_norm
