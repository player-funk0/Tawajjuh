"""
train_torch.py -- Trains the PyTorch reimplementation (network_torch.py)
on the exact same bilingual dataset (dataset.py) used for the numpy
version, so results are directly comparable.

===========================================================================
RUN THIS YOURSELF TO VERIFY -- it could not be executed in the sandbox
this project was built in (no network access to install torch there).
===========================================================================

    pip install torch
    python train_torch.py

WHAT TO CHECK (a 30-second sanity check, no ML expertise needed):
  1. Loss should decrease steadily and not explode to NaN/inf.
  2. Final validation accuracy should land in a similar range to the
     numpy version's reported numbers (report.md Section 8): ~97-100% for
     both English and Egyptian Arabic.
  3. The per-language and per-source (template vs hand-written) breakdown
     should show no language/source collapsing to near-chance accuracy --
     if one does, something is wrong (e.g. a tokenization mismatch).

If those three hold, the PyTorch version is behaving consistently with
the verified numpy version, which is what "correct" looks like here
given the two were never expected to produce bit-identical numbers
(different optimizer, multi-head vs single-head attention, different
random initialization).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import build_dataset
from network_torch import TransformerIntentClassifierTorch

SEED = 42
MAX_LEN = 12
D_MODEL = 32
NUM_HEADS = 4
HIDDEN = 32
EPOCHS = 150
LR = 1e-3  # Adam typically wants a much smaller LR than the plain SGD used in train.py


def train_val_split(n, val_fraction=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    return idx[n_val:], idx[:n_val]


def main():
    torch.manual_seed(SEED)

    print("Building dataset (identical pipeline to the numpy version)...")
    data = build_dataset(max_len=MAX_LEN, seed=SEED)
    train_idx, val_idx = train_val_split(len(data["y"]), 0.2, seed=SEED)

    X_ids = torch.tensor(data["X_ids"], dtype=torch.long)
    X_mask = torch.tensor(data["X_mask"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)

    Xtr_ids, Xtr_mask, ytr = X_ids[train_idx], X_mask[train_idx], y[train_idx]
    Xval_ids, Xval_mask, yval = X_ids[val_idx], X_mask[val_idx], y[val_idx]

    model = TransformerIntentClassifierTorch(
        vocab_size=len(data["vocab"]), max_len=MAX_LEN,
        d_model=D_MODEL, num_heads=NUM_HEADS, hidden=HIDDEN,
        num_classes=len(data["intents"]),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print("\nTraining with Adam + multi-head attention (4 heads)...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(Xtr_ids, Xtr_mask)
        loss = criterion(logits, ytr)
        loss.backward()
        optimizer.step()

        if epoch % 15 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_acc = (logits.argmax(dim=1) == ytr).float().mean().item()
                val_logits, _ = model(Xval_ids, Xval_mask)
                val_acc = (val_logits.argmax(dim=1) == yval).float().mean().item()
            print(f"Epoch {epoch:4d} | loss {loss.item():.4f} | train acc {train_acc*100:5.1f}% | val acc {val_acc*100:5.1f}%")

    model.eval()
    with torch.no_grad():
        val_logits, _ = model(Xval_ids, Xval_mask)
        val_preds = val_logits.argmax(dim=1)
        val_acc = (val_preds == yval).float().mean().item()

    print(f"\nFinal validation accuracy: {val_acc*100:.1f}%")

    val_langs = data["langs"][val_idx]
    val_sources = data["sources"][val_idx]
    val_preds_np = val_preds.numpy()
    yval_np = yval.numpy()

    print("\nPer-language breakdown:")
    for lang, label in [("en", "English"), ("ar", "Egyptian Arabic")]:
        m = val_langs == lang
        if m.sum() > 0:
            acc = (val_preds_np[m] == yval_np[m]).mean()
            print(f"  {label}: {acc*100:.1f}% (n={m.sum()})")

    print("\nPer-source breakdown:")
    for source, label in [("template", "Template-generated"), ("hand_written", "Hand-written")]:
        m = val_sources == source
        if m.sum() > 0:
            acc = (val_preds_np[m] == yval_np[m]).mean()
            print(f"  {label}: {acc*100:.1f}% (n={m.sum()})")

    torch.save({
        "model_state": model.state_dict(),
        "vocab_words": data["vocab"].words,
        "intents": data["intents"],
        "max_len": MAX_LEN, "d_model": D_MODEL, "num_heads": NUM_HEADS, "hidden": HIDDEN,
    }, "trained_model_torch.pt")
    print("\nSaved trained_model_torch.pt")


if __name__ == "__main__":
    main()
