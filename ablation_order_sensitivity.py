"""
ablation_order_sensitivity.py -- Runs the diagnostic experiment described
in synthetic_order_task.py: trains the FULL attention model against a
strictly order-blind baseline (no attention, no positional encoding) and
Naive Bayes, on a task where word order is the ONLY signal.

This directly tests the theoretical claim in report.md Section 3.4/5
empirically, using a task specifically constructed so the order-blind
models have a provable 50% ceiling (see synthetic_order_task.py for the
information-theoretic argument).
"""

import numpy as np
import matplotlib.pyplot as plt

from synthetic_order_task import generate_order_task, verify_bow_uninformative
from dataset import Vocab
from network import TransformerIntentClassifier, OrderBlindClassifier
from baseline_naive_bayes import NaiveBayesClassifier

SEED = 42
SEQ_LEN = 4
MAX_LEN = 4
N_TRAIN = 400
N_VAL = 150
EPOCHS = 300
LR = 0.3


def encode_all(texts, vocab, max_len):
    ids, masks = [], []
    for t in texts:
        i, m = vocab.encode(t, max_len)
        ids.append(i)
        masks.append(m)
    return np.array(ids), np.array(masks)


def train_model(model, X_ids, X_mask, y, epochs, lr):
    history = []
    for epoch in range(1, epochs + 1):
        logits = model.forward(X_ids, X_mask)
        loss = model.compute_loss(logits, y)
        model.backward()
        model.step(lr)
        if epoch % 20 == 0:
            acc = (logits.argmax(axis=1) == y).mean()
            history.append((epoch, loss, acc))
    return history


def main():
    print("=" * 60)
    print("Sanity-checking the theoretical claim on the generated data")
    print("=" * 60)
    train_texts, y_train = generate_order_task(N_TRAIN, SEQ_LEN, seed=SEED)
    val_texts, y_val = generate_order_task(N_VAL, SEQ_LEN, seed=SEED + 1)
    max_dev = verify_bow_uninformative(train_texts + val_texts, np.concatenate([y_train, y_val]))
    print(f"Max filler-token/label correlation: {max_dev:.4f} (near-zero confirms bag-of-words "
          f"has no usable signal; label depends purely on MARK_X/MARK_Y order)")

    vocab = Vocab(train_texts + val_texts)
    Xtr_ids, Xtr_mask = encode_all(train_texts, vocab, MAX_LEN)
    Xval_ids, Xval_mask = encode_all(val_texts, vocab, MAX_LEN)

    print("\n" + "=" * 60)
    print("Training FULL attention model (attention + positional encoding)")
    print("=" * 60)
    attn_model = TransformerIntentClassifier(
        vocab_size=len(vocab), max_len=MAX_LEN, d_model=16, d_k=16, hidden=16,
        num_classes=2, seed=SEED,
    )
    attn_history = train_model(attn_model, Xtr_ids, Xtr_mask, y_train, EPOCHS, LR)
    attn_val_acc = (attn_model.forward(Xval_ids, Xval_mask).argmax(axis=1) == y_val).mean()
    print(f"Full attention model validation accuracy: {attn_val_acc*100:.1f}%")

    print("\n" + "=" * 60)
    print("Training ORDER-BLIND baseline (no attention, no positional encoding)")
    print("=" * 60)
    blind_model = OrderBlindClassifier(vocab_size=len(vocab), d_model=16, hidden=16, num_classes=2, seed=SEED)
    blind_history = train_model(blind_model, Xtr_ids, Xtr_mask, y_train, EPOCHS, LR)
    blind_val_acc = (blind_model.forward(Xval_ids, Xval_mask).argmax(axis=1) == y_val).mean()
    print(f"Order-blind baseline validation accuracy: {blind_val_acc*100:.1f}%  "
          f"(theoretical ceiling: 50%)")

    print("\n" + "=" * 60)
    print("Training Naive Bayes (also inherently bag-of-words)")
    print("=" * 60)
    nb = NaiveBayesClassifier(vocab_size=len(vocab), num_classes=2)
    nb.fit(Xtr_ids, Xtr_mask, y_train)
    nb_val_acc = (nb.predict(Xval_ids, Xval_mask) == y_val).mean()
    print(f"Naive Bayes validation accuracy: {nb_val_acc*100:.1f}%  (theoretical ceiling: 50%)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<35}{'Val Accuracy':>15}")
    print(f"{'Full attention (order-aware)':<35}{attn_val_acc*100:>14.1f}%")
    print(f"{'Order-blind neural baseline':<35}{blind_val_acc*100:>14.1f}%")
    print(f"{'Naive Bayes':<35}{nb_val_acc*100:>14.1f}%")
    print("\nInterpretation: the order-blind models cluster near the 50% chance")
    print("ceiling predicted by the information-theoretic argument, while the")
    print("attention model -- the only one with access to positional information")
    print("-- solves the task. This is direct empirical evidence for the")
    print("permutation-invariance claim in report.md Section 3.4/5, on a task")
    print("specifically designed so that claim is the only thing being measured.")

    # Bar chart for the report
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["Attention\n(order-aware)", "Order-blind\nneural baseline", "Naive Bayes"]
    accs = [attn_val_acc * 100, blind_val_acc * 100, nb_val_acc * 100]
    colors = ["#2b7a78", "#c94c4c", "#c9954c"]
    bars = ax.bar(names, accs, color=colors)
    ax.axhline(50, color="gray", linestyle="--", label="Theoretical chance ceiling (50%)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Order-Sensitivity Ablation: Can the Model Use Word Order?")
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 2, f"{acc:.1f}%", ha="center")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ablation_order_sensitivity.png", dpi=150)
    print("\nSaved ablation_order_sensitivity.png")


if __name__ == "__main__":
    main()
