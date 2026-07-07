"""
baseline_naive_bayes.py -- Multinomial Naive Bayes, derived and implemented
from scratch (no sklearn), as a theoretically-grounded baseline to compare
the attention model against.

===========================================================================
THE MATH
===========================================================================

Naive Bayes applies Bayes' theorem with a conditional-independence
assumption between features (words), given the class:

    P(c | doc) proportional-to  P(c) * prod_i P(w_i | c)

Taking logs for numerical stability (avoids underflow from multiplying many
small probabilities) turns the product into a sum:

    log P(c | doc) = log P(c) + sum_i count(w_i, doc) * log P(w_i | c) + const

Parameters are estimated by Maximum Likelihood with Laplace (add-1)
smoothing, which corresponds to a Bayesian MAP estimate under a symmetric
Dirichlet prior over word probabilities -- this is what prevents zero
probabilities for words unseen in a given class during training:

    P(w | c) = (count(w, c) + 1) / (total_words(c) + |Vocab|)

This model assumes word ORDER doesn't matter (bag-of-words) and words are
independent given the class -- both are false in general language, which is
exactly the theoretical weakness the attention model is designed to
address. Comparing the two is a direct empirical test of that theory.
"""

import numpy as np


class NaiveBayesClassifier:
    def __init__(self, vocab_size, num_classes, alpha=1.0):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.alpha = alpha  # Laplace smoothing strength
        self.log_prior = None
        self.log_likelihood = None  # [num_classes, vocab_size]

    def _to_bow(self, X_ids, X_mask):
        """Convert padded token-id sequences into bag-of-words count vectors."""
        N = X_ids.shape[0]
        bow = np.zeros((N, self.vocab_size))
        for i in range(N):
            for tok, m in zip(X_ids[i], X_mask[i]):
                if m > 0:
                    bow[i, tok] += 1
        return bow

    def fit(self, X_ids, X_mask, y):
        bow = self._to_bow(X_ids, X_mask)
        self.log_prior = np.zeros(self.num_classes)
        self.log_likelihood = np.zeros((self.num_classes, self.vocab_size))

        for c in range(self.num_classes):
            docs_c = bow[y == c]
            self.log_prior[c] = np.log(len(docs_c) / len(bow))
            word_counts = docs_c.sum(axis=0) + self.alpha
            self.log_likelihood[c] = np.log(word_counts / word_counts.sum())

    def predict(self, X_ids, X_mask):
        bow = self._to_bow(X_ids, X_mask)
        # log P(c|doc) ~ log_prior + bow @ log_likelihood^T
        scores = self.log_prior[None, :] + bow @ self.log_likelihood.T
        return scores.argmax(axis=1)
