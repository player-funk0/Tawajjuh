"""
train.py -- Full experimental pipeline.

1. Build dataset, split train/validation.
2. Sanity-check the hand-derived backward pass against numerical gradients
   (this is the single most important engineering step when implementing
   backprop by hand -- silent gradient bugs are the most common source of
   "the model trains but is subtly wrong").
3. Train the attention model with plain SGD.
4. Train the Naive Bayes baseline for comparison.
5. Plot loss/accuracy curves, confusion matrix, and an attention-weight
   heatmap for interpretability.
"""

import numpy as np
import matplotlib.pyplot as plt

from dataset import build_dataset
from network import TransformerIntentClassifier
from baseline_naive_bayes import NaiveBayesClassifier
from core import numerical_gradient, clip_grad_norm

SEED = 42
MAX_LEN = 12
D_MODEL = 24
D_K = 24
HIDDEN = 24
EPOCHS = 300
LR = 0.3
LR_DECAY = 0.995        # lr_epoch = LR * LR_DECAY ** epoch
MAX_GRAD_NORM = 5.0     # global gradient-norm clipping threshold
VAL_FRACTION = 0.2


def train_val_split(data, val_fraction=0.2, seed=0):
    n = len(data["y"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return train_idx, val_idx


def gradient_check(model, X_ids, X_mask, y, n_checks=20):
    """
    Compares hand-derived analytical gradients against numerical gradients
    on a small batch. If these don't match closely, there's a bug in the
    backward() derivation -- this is the standard way to catch that before
    trusting any training results.
    """
    logits = model.forward(X_ids, X_mask)
    model.compute_loss(logits, y)
    model.backward()

    def loss_only():
        logits = model.forward(X_ids, X_mask)
        return model.compute_loss(logits, y)

    max_rel_error = 0.0
    checked = 0
    for name, param, analytical_grad in model.all_params_flat():
        if checked >= n_checks:
            break
        flat_param = param.reshape(-1)
        flat_grad = analytical_grad.reshape(-1)
        # Sample a few random entries per parameter tensor rather than all
        # (full numerical gradient checking is O(num_params) forward passes,
        # too slow to do exhaustively, but a random sample is a valid and
        # standard way to catch bugs).
        sample_idxs = np.random.default_rng(0).choice(len(flat_param), size=min(3, len(flat_param)), replace=False)
        for idx in sample_idxs:
            orig_shape = param.shape
            flat_view = param.reshape(-1)

            def f():
                return loss_only()

            eps = 1e-5
            orig = flat_view[idx]
            flat_view[idx] = orig + eps
            f_plus = f()
            flat_view[idx] = orig - eps
            f_minus = f()
            flat_view[idx] = orig
            numerical = (f_plus - f_minus) / (2 * eps)
            analytical = flat_grad[idx]
            rel_error = abs(numerical - analytical) / (abs(numerical) + abs(analytical) + 1e-8)
            max_rel_error = max(max_rel_error, rel_error)
            checked += 1

    return max_rel_error


def main():
    print("=" * 60)
    print("1. Building dataset")
    print("=" * 60)
    data = build_dataset(max_len=MAX_LEN, seed=SEED)
    train_idx, val_idx = train_val_split(data, VAL_FRACTION, seed=SEED)
    print(f"Total examples: {len(data['y'])} | Train: {len(train_idx)} | Val: {len(val_idx)}")
    print(f"Vocab size: {len(data['vocab'])}")

    Xtr_ids, Xtr_mask, ytr = data["X_ids"][train_idx], data["X_mask"][train_idx], data["y"][train_idx]
    Xval_ids, Xval_mask, yval = data["X_ids"][val_idx], data["X_mask"][val_idx], data["y"][val_idx]

    print("\n" + "=" * 60)
    print("2. Gradient checking (verifying hand-derived backprop)")
    print("=" * 60)
    model = TransformerIntentClassifier(
        vocab_size=len(data["vocab"]), max_len=MAX_LEN,
        d_model=D_MODEL, d_k=D_K, hidden=HIDDEN,
        num_classes=len(data["intents"]), seed=SEED,
    )
    small_batch = slice(0, 8)
    max_rel_error = gradient_check(model, Xtr_ids[small_batch], Xtr_mask[small_batch], ytr[small_batch])
    print(f"Max relative error (analytical vs numerical gradient): {max_rel_error:.2e}")
    print("(Values below ~1e-4 indicate the backward pass is mathematically correct)")
    assert max_rel_error < 1e-3, "Gradient check failed -- backward pass has a bug"
    print("Gradient check PASSED.")

    print("\n" + "=" * 60)
    print("3. Training attention model (plain SGD, full-batch)")
    print("=" * 60)
    model = TransformerIntentClassifier(
        vocab_size=len(data["vocab"]), max_len=MAX_LEN,
        d_model=D_MODEL, d_k=D_K, hidden=HIDDEN,
        num_classes=len(data["intents"]), seed=SEED,
    )
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": []}
    grad_norm_history = {"epoch": [], "raw_norm": [], "lr": []}
    for epoch in range(1, EPOCHS + 1):
        logits = model.forward(Xtr_ids, Xtr_mask)
        loss = model.compute_loss(logits, ytr)
        model.backward()

        current_lr = LR * (LR_DECAY ** epoch)
        raw_norm = clip_grad_norm(model.all_params_flat(), MAX_GRAD_NORM)
        model.step(current_lr)

        grad_norm_history["epoch"].append(epoch)
        grad_norm_history["raw_norm"].append(raw_norm)
        grad_norm_history["lr"].append(current_lr)

        if epoch % 10 == 0 or epoch == 1:
            train_acc = (logits.argmax(axis=1) == ytr).mean()
            val_logits = model.forward(Xval_ids, Xval_mask)
            val_acc = (val_logits.argmax(axis=1) == yval).mean()
            history["epoch"].append(epoch)
            history["train_loss"].append(loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            print(f"Epoch {epoch:4d} | lr {current_lr:.4f} | grad_norm {raw_norm:6.2f} | "
                  f"loss {loss:.4f} | train acc {train_acc*100:5.1f}% | val acc {val_acc*100:5.1f}%")

    print("\n" + "=" * 60)
    print("4. Training Naive Bayes baseline")
    print("=" * 60)
    nb = NaiveBayesClassifier(vocab_size=len(data["vocab"]), num_classes=len(data["intents"]))
    nb.fit(Xtr_ids, Xtr_mask, ytr)
    nb_val_preds = nb.predict(Xval_ids, Xval_mask)
    nb_val_acc = (nb_val_preds == yval).mean()
    print(f"Naive Bayes validation accuracy: {nb_val_acc*100:.1f}%")

    attn_val_logits = model.forward(Xval_ids, Xval_mask)
    attn_val_preds = attn_val_logits.argmax(axis=1)
    attn_val_acc = (attn_val_preds == yval).mean()
    print(f"Attention model validation accuracy: {attn_val_acc*100:.1f}%")

    # Honest per-language breakdown -- an aggregate accuracy number can
    # hide a model that's only really working in one language.
    val_langs = data["langs"][val_idx]
    for lang, label in [("en", "English"), ("ar", "Egyptian Arabic")]:
        lang_mask = val_langs == lang
        if lang_mask.sum() > 0:
            lang_acc = (attn_val_preds[lang_mask] == yval[lang_mask]).mean()
            print(f"  -> {label} validation accuracy: {lang_acc*100:.1f}% (n={lang_mask.sum()})")

    # Honest check on dataset realism: does accuracy hold up on the
    # hand-written, non-templated examples specifically, or only on the
    # cleaner template-generated ones? This is the direct empirical
    # answer to "does template data overstate real-world generalization."
    val_sources = data["sources"][val_idx]
    for source, label in [("template", "Template-generated"), ("hand_written", "Hand-written (messy)")]:
        src_mask = val_sources == source
        if src_mask.sum() > 0:
            src_acc = (attn_val_preds[src_mask] == yval[src_mask]).mean()
            print(f"  -> {label} validation accuracy: {src_acc*100:.1f}% (n={src_mask.sum()})")

    print("\n" + "=" * 60)
    print("5. Saving diagnostics (plots + weights)")
    print("=" * 60)

    # Loss/accuracy curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss"); axes[0].set_title("Training loss")
    axes[1].plot(history["epoch"], history["train_acc"], label="train")
    axes[1].plot(history["epoch"], history["val_acc"], label="validation")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Saved training_curves.png")

    # Gradient-norm / learning-rate diagnostic: direct evidence for
    # whether clipping + decay are doing anything, rather than just
    # asserting they help.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(grad_norm_history["epoch"], grad_norm_history["raw_norm"])
    axes[0].axhline(MAX_GRAD_NORM, color="red", linestyle="--", label=f"clip threshold ({MAX_GRAD_NORM})")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Global gradient norm (pre-clip)")
    axes[0].set_title("Gradient norm per epoch"); axes[0].legend()
    axes[1].plot(grad_norm_history["epoch"], grad_norm_history["lr"])
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Learning rate")
    axes[1].set_title(f"LR decay schedule (LR0={LR}, decay={LR_DECAY})")
    plt.tight_layout()
    plt.savefig("gradient_diagnostics.png", dpi=150)
    print("Saved gradient_diagnostics.png")
    n_clipped = sum(1 for n in grad_norm_history["raw_norm"] if n > MAX_GRAD_NORM)
    print(f"Gradient clipping engaged on {n_clipped}/{EPOCHS} epochs "
          f"(max observed norm: {max(grad_norm_history['raw_norm']):.2f})")

    # Confusion matrix
    num_classes = len(data["intents"])
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_c, pred_c in zip(yval, attn_val_preds):
        cm[true_c, pred_c] += 1
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks(range(num_classes)); ax.set_xticklabels(data["intents"])
    ax.set_yticks(range(num_classes)); ax.set_yticklabels(data["intents"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (attention model)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

    # Attention heatmap for one example (interpretability)
    example_idx = val_idx[0]
    ids = data["X_ids"][example_idx:example_idx+1]
    mask = data["X_mask"][example_idx:example_idx+1]
    model.forward(ids, mask)
    attn_weights = model.attn.attention_weights()[0]  # [N, N]
    n_tokens = int(mask.sum())
    tokens = [data["vocab"].words[t] for t in ids[0][:n_tokens]]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(attn_weights[:n_tokens, :n_tokens], cmap="viridis")
    ax.set_xticks(range(n_tokens)); ax.set_xticklabels(tokens, rotation=45)
    ax.set_yticks(range(n_tokens)); ax.set_yticklabels(tokens)
    ax.set_title(f"Attention weights for: \"{data['texts'][example_idx]}\"")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    print("Saved attention_heatmap.png")

    # Save trained model weights + vocab for the assistant to use
    np.savez(
        "trained_model.npz",
        embedding_table=model.embedding.table,
        Wq=model.attn.Wq, Wk=model.attn.Wk, Wv=model.attn.Wv,
        dense1_W=model.dense1.W, dense1_b=model.dense1.b,
        dense2_W=model.dense2.W, dense2_b=model.dense2.b,
        vocab_words=np.array(data["vocab"].words),
        intents=np.array(data["intents"]),
        max_len=MAX_LEN, d_model=D_MODEL, d_k=D_K, hidden=HIDDEN,
    )
    print("Saved trained_model.npz")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Gradient check max relative error : {max_rel_error:.2e}  (PASSED)")
    print(f"Naive Bayes validation accuracy    : {nb_val_acc*100:.1f}%")
    print(f"Attention model validation accuracy: {attn_val_acc*100:.1f}%")


if __name__ == "__main__":
    main()
