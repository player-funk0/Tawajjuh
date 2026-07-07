"""
synthetic_order_task.py -- A controlled diagnostic experiment, separate
from the main app/file dataset, built specifically to test one precise
theoretical claim from report.md Section 3.4/5: that self-attention +
positional encoding can use word ORDER, while a bag-of-words-style model
mathematically cannot.

===========================================================================
WHY THE MAIN DATASET CAN'T TEST THIS CLEANLY
===========================================================================

In the real app/file dataset, intent is mostly determined by WHICH words
appear (file nouns vs app names), so a bag-of-words model can do well
without ever using order -- which is exactly what Section 5 found (Naive
Bayes ties the attention model there). That's an honest result about that
task, but it means that dataset cannot demonstrate attention's theoretical
advantage, because the task doesn't require it. Testing a claim requires a
task where the claim actually matters.

===========================================================================
THE DIAGNOSTIC TASK
===========================================================================

Each example is a sequence of 4 tokens. Two of the four positions contain
fixed marker tokens "MARK_X" and "MARK_Y" (always exactly one of each);
the other two positions contain filler tokens drawn uniformly at random
from a small filler vocabulary, INDEPENDENT of the label. The label is:

    label = 1   if MARK_X appears before MARK_Y in the sequence
    label = 0   otherwise

===========================================================================
WHY THIS PROVES THE CLAIM MATHEMATICALLY, NOT JUST EMPIRICALLY
===========================================================================

By construction, every example contains EXACTLY one MARK_X and one
MARK_Y -- so the bag-of-words count vector for these two tokens is the
constant (1, 1) for every single example, regardless of label. Filler
token counts are drawn independently of the label by construction. This
means the bag-of-words representation of an example carries ZERO mutual
information about its label:

    I(bag_of_words(seq) ; label) = 0

Therefore ANY model that only sees word counts/presence (Naive Bayes, or
a neural network using mean-pooled embeddings with no positional
information) cannot exceed 50% accuracy in expectation on this task --
not "tends not to," but literally cannot, by the information-theoretic
argument above. This gives us a hard, provable floor to check the
OrderBlindClassifier baseline against, rather than an empirical
"it did worse" result that could be attributed to other factors (capacity,
training time, etc).

A model that CAN use position (attention + positional encoding) has no
such ceiling, since MARK_X's and MARK_Y's positions differ across
examples and directly determine the label.
"""

import numpy as np

FILLERS = ["dog", "cat", "sun", "moon", "rock", "leaf", "cloud", "river"]
MARK_X = "markx"
MARK_Y = "marky"


def generate_order_task(n_examples=400, seq_len=4, seed=0):
    rng = np.random.default_rng(seed)
    texts, labels = [], []

    for _ in range(n_examples):
        pos_x, pos_y = rng.choice(seq_len, size=2, replace=False)
        tokens = [None] * seq_len
        tokens[pos_x] = MARK_X
        tokens[pos_y] = MARK_Y
        for i in range(seq_len):
            if tokens[i] is None:
                tokens[i] = rng.choice(FILLERS)  # label-independent filler
        texts.append(" ".join(tokens))
        labels.append(1 if pos_x < pos_y else 0)

    return texts, np.array(labels, dtype=np.int64)


def verify_bow_uninformative(texts, labels):
    """
    Sanity-check the theoretical claim directly on the generated data:
    confirms that MARK_X and MARK_Y counts are constant across all
    examples (so bag-of-words truly carries no label signal), and that
    filler token distribution doesn't correlate with the label either.
    """
    from collections import Counter

    for t in texts:
        words = t.split()
        assert words.count(MARK_X) == 1 and words.count(MARK_Y) == 1

    # Check filler words aren't accidentally correlated with label
    filler_counts_by_label = {0: Counter(), 1: Counter()}
    for t, y in zip(texts, labels):
        for w in t.split():
            if w in FILLERS:
                filler_counts_by_label[int(y)][w] += 1

    total0 = sum(filler_counts_by_label[0].values())
    total1 = sum(filler_counts_by_label[1].values())
    max_dev = 0.0
    for w in FILLERS:
        p0 = filler_counts_by_label[0][w] / total0
        p1 = filler_counts_by_label[1][w] / total1
        max_dev = max(max_dev, abs(p0 - p1))

    return max_dev  # should be small (statistical noise only, no true correlation)


if __name__ == "__main__":
    texts, labels = generate_order_task(n_examples=20, seed=1)
    for t, y in zip(texts[:10], labels[:10]):
        print(f"{t:30s} -> label={y}")
    dev = verify_bow_uninformative(*generate_order_task(n_examples=2000, seed=2))
    print(f"\nMax filler/label correlation deviation (should be small, noise only): {dev:.4f}")
