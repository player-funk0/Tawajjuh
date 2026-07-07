# A From-Scratch Attention-Based Intent Classifier for Device Command Recognition

## Abstract

This project implements a single-head self-attention neural network entirely
from raw linear algebra (numpy, no autograd), applies it to the task of
classifying natural-language device commands, and rigorously verifies the
hand-derived backpropagation via numerical gradient checking. The model is
compared against a classical, theoretically-grounded Naive Bayes baseline
also implemented from scratch. The system was subsequently extended to
understand both English and Egyptian Arabic within a single shared model,
and wrapped with an offline voice interface (speech-to-text and
text-to-speech). The goal throughout is not to maximize accuracy on a toy
task, but to demonstrate correct, verified understanding of the mathematics
underlying modern attention-based architectures, and to report results
honestly, including where a simpler baseline performs comparably and where
engineering tradeoffs (like offline dialectal speech recognition) genuinely
limit real-world accuracy.

## 1. Problem Setup

Given a natural-language command (e.g. "open notepad", "show me my resume"),
predict:
1. **Intent** -- OPEN_APP or OPEN_FILE (the learned classification task)
2. **Target** -- which app/file (extracted via a simple, deterministic
   rule, deliberately kept outside the learned model; see Section 6)

The dataset (`dataset.py`) is template-generated: 22 app/file names across
24 phrasing templates, including templates where the target word appears in
different positions ("open X" vs "X please open"), producing 220 labeled
examples. This is intentional: a model that only reads word *presence*
(bag-of-words) cannot distinguish some of these cases from their intended
meaning as well as a model that can use word *order* -- creating a fair,
motivated testbed for attention and positional encoding.

## 2. Model Architecture

```
tokens (padded to length 10)
   -> Embedding (learned, vocab_size x 24)
   -> + sinusoidal positional encoding (fixed, not learned)
   -> Self-Attention (single head, d_k = 24)
   -> masked mean-pool over sequence
   -> Dense(24 -> 24) + ReLU
   -> Dense(24 -> 2)
   -> Softmax + Cross-Entropy loss
```

Total learned parameters: embedding table + 3 attention projection matrices
+ 2 dense layers -- on the order of a few thousand parameters. This is
intentionally small: the point is mathematical correctness and
interpretability, not scale.

## 3. The Core Mathematics

### 3.1 Scaled dot-product self-attention

For token embeddings $X \in \mathbb{R}^{N \times d}$:

$$Q = XW_Q,\quad K = XW_K,\quad V = XW_V$$
$$S = \frac{QK^\top}{\sqrt{d_k}}, \quad A = \text{softmax}(S), \quad O = AV$$

The $1/\sqrt{d_k}$ scaling controls the variance of the pre-softmax scores:
for random $Q,K$ with unit-variance components, $\text{Var}(Q_i \cdot K_j) =
d_k$, and without correction this pushes softmax into saturated,
low-gradient regions as $d_k$ grows. Dividing by $\sqrt{d_k}$ keeps the
variance approximately constant regardless of dimensionality (Vaswani et
al., 2017).

### 3.2 Backpropagation through attention (hand-derived)

Given $\partial L/\partial O$ from the next layer:

$$\frac{\partial L}{\partial V} = A^\top \frac{\partial L}{\partial O}, \qquad
  \frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} V^\top$$

Backprop through the row-wise softmax uses the identity (for a single row
$a = \text{softmax}(s)$ and upstream gradient $g = \partial L/\partial a$):

$$\frac{\partial L}{\partial s} = a \odot \left(g - \sum_j a_j g_j\right)$$

which follows from the softmax Jacobian $\partial a_i/\partial s_j = a_i(\delta_{ij}
- a_j)$ contracted with $g$. Then, through the scaled dot product:

$$\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}}\frac{\partial L}{\partial S}K,
  \qquad
  \frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}}\frac{\partial L}{\partial S}^\top Q$$

and finally through the linear projections via the standard matrix-calculus
rule for $Y = XW$: $\partial L/\partial W = X^\top (\partial L/\partial Y)$,
$\partial L/\partial X = (\partial L/\partial Y)W^\top$, summed across the
three projections since $X$ feeds all of $Q, K, V$.

### 3.3 Why cross-entropy is the right loss (information-theoretic view)

Cross-entropy loss $L = -\sum_i y_i \log p_i$ is the negative
log-likelihood of the true label under the model's predicted distribution.
For a one-hot true label $y$, this equals the KL divergence between $y$ and
the predicted distribution $p$:

$$D_{KL}(y \| p) = \sum_i y_i \log\frac{y_i}{p_i} = \underbrace{-H(y)}_{=0
\text{ for one-hot } y} + \text{CrossEntropy}(y, p)$$

So training minimizes the information-theoretic divergence between the
model's beliefs and the true label distribution. This also explains why the
combined softmax+cross-entropy gradient with respect to the logits
simplifies to the clean form $\partial L/\partial z = p - y$: the softmax
Jacobian and the log-likelihood gradient cancel algebraically.

### 3.4 Positional encoding and permutation invariance

Self-attention as defined is provably permutation-equivariant: permuting
the rows of $X$ produces the same permutation of $O$, with no change to
the attention weights themselves. Without additional information, the
model literally cannot represent word order. We inject order via fixed
sinusoidal positional encodings:

$$PE(pos, 2i) = \sin(pos / 10000^{2i/d}), \quad PE(pos, 2i{+}1) = \cos(pos / 10000^{2i/d})$$

Each dimension oscillates at a different frequency, giving every position a
unique, smoothly-varying signature that generalizes to unseen sequence
lengths (unlike learned absolute position embeddings).

## 4. Verifying Correctness: Gradient Checking

Hand-derived backpropagation is easy to get subtly wrong -- a transposed
matrix or missing sum can silently produce a model that trains but
converges to the wrong optimum, or trains slower than it should. Before
trusting any results, `train.py` runs **numerical gradient checking**:
for randomly sampled parameters, it compares the analytical gradient
(from `backward()`) against a central-difference numerical estimate:

$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i + \epsilon) - L(\theta_i -
\epsilon)}{2\epsilon}$$

**Result:** maximum relative error across sampled parameters was on the
order of $10^{-7}$, well below the $10^{-4}$ threshold conventionally used
to certify a backward pass as correct. This is the standard verification
technique used in research-grade implementations before any training run
is trusted.

## 5. Experimental Results

| Model | Validation Accuracy |
|---|---|
| Naive Bayes (bag-of-words, from scratch) | ~100% |
| Self-attention model (from scratch) | ~97-100%, converges by ~epoch 100 |

**Honest finding:** on this dataset, the Naive Bayes baseline matches or
slightly exceeds the attention model. This is not a failure of the
attention model -- it's an expected and informative result. The dataset is
small (220 examples) and the intent distinction (OPEN_APP vs OPEN_FILE) is
largely determined by *which words appear* (file extensions, "file",
"document") rather than subtle word-order effects, so the bag-of-words
assumption underlying Naive Bayes is not badly violated here. The
attention model's theoretical advantage -- using word order and context --
would be expected to matter more on a harder task with more genuine
ambiguity. Reporting this honestly, rather than only showing favorable
numbers, is itself part of doing this correctly: a strong project shows
you understand *why* a result came out the way it did, not just that a
number is high.

Training and diagnostic artifacts (generated by `train.py`):
- `training_curves.png` -- loss and accuracy over training
- `confusion_matrix.png` -- per-class error breakdown
- `attention_heatmap.png` -- visualizes which tokens the model attends to
  for a sample command, providing a degree of interpretability rarely
  available in bag-of-words models

## 6. Design Decisions and Honest Limitations

- **Target extraction is rule-based, not learned.** Given the tiny
  dataset, training a second model (e.g. sequence tagging / named entity
  recognition) to extract the target word would be data-hungry and hard to
  validate rigorously. The rule-based approach is simple, auditable, and
  correctly scoped to the amount of data available -- a deliberate
  engineering tradeoff, not an oversight. Section 7 discusses how to
  extend this properly.
- **Single attention head, one layer.** This is sufficient to demonstrate
  and validate the mechanism's mathematics. Real transformers stack many
  layers and heads; doing so here would add engineering complexity without
  adding mathematical insight for this task.
- **Plain SGD, not Adam.** Full-batch gradient descent was sufficient to
  converge on this small dataset. Adam (with its bias-corrected first/second
  moment estimates) is a natural extension for noisier, larger-batch
  settings, and is a documented next step rather than a needed fix here.

## 7. Future Extensions

1. Multi-head attention -- multiple parallel attention computations with
   different learned projections, concatenated, letting the model attend
   to different types of relationships simultaneously.
2. Learned slot-filling (NER-style tagging) to replace the rule-based
   target extractor, evaluated with precision/recall rather than accuracy
   alone.
3. A harder, more ambiguous dataset (e.g. commands where word order
   changes the correct intent) to create a setting where the attention
   model's theoretical advantage over bag-of-words should empirically
   manifest -- directly testing the hypothesis in Section 5.
4. Replacing SGD with Adam and comparing convergence speed, with a
   derivation of Adam's bias-correction terms.

## 8. Extension: Bilingual (English + Egyptian Arabic) Understanding

The dataset and tokenizer were extended to support Egyptian Arabic
alongside English, training a single shared classifier rather than two
separate models.

**Tokenizer fix.** The original tokenizer regex (`[a-z0-9]+`) silently
dropped every Arabic character, since Arabic script (Unicode block
U+0600-U+06FF) falls outside that character class. The fix adds an
alternation: `[a-z0-9]+|[\u0600-\u06FF]+`. This is a small change with a
large consequence -- without it, Arabic input would tokenize to an empty
sequence and the model would receive no signal at all, not just weak
signal.

**Bilingual training data.** A lexicon (`lexicon.py`) maps each canonical
app/file target to its surface forms in both languages (e.g. `chrome` ->
`["chrome", "كروم", "الكروم", "جوجل كروم"]`), including the Arabic
definite article ("ال") attached directly to the noun, as it appears in
real usage. Templates in both languages generate the training set, so the
model learns a single shared embedding space where "افتح" and "open" play
the same functional role -- purely from co-occurring with the same
intent label during training, not from any hand-coded translation.

**Honest result:** validation accuracy was measured *separately* per
language rather than only in aggregate, since an aggregate number can
hide a model that secretly only works in one language:

| Language | Validation Accuracy |
|---|---|
| English | ~97% |
| Egyptian Arabic | ~100% |

The gradient check (Section 4) was re-run after this change and still
passed (max relative error ~1e-8), confirming the backward pass remains
mathematically correct after the tokenizer and dataset changes -- this is
exactly the kind of re-verification that should happen after any change
to the input pipeline, not just at initial implementation.

**Known simplification:** Egyptian Arabic templates do not enforce
grammatical gender/number agreement across all surface forms (e.g. object
pronouns attached to verbs). This is a deliberate scope decision for a
template-generated dataset, not an oversight -- full agreement would
require a real morphological analyzer, which is its own research area
separate from the attention mechanism this project is about.

## 9. Extension: Voice Input/Output

Voice was added as an I/O layer around the existing text classifier,
using two purpose-built offline tools rather than attempting to train
speech recognition or synthesis from scratch:

- **Speech-to-text:** Vosk, an offline toolkit, using separate small
  acoustic models for English and Arabic.
- **Text-to-speech:** pyttsx3, which drives the voices already installed
  in Windows.

This is a deliberate and honest architectural choice worth stating
explicitly: **training a competitive speech recognition or speech
synthesis model from scratch is not feasible for a project at this
scale** -- production ASR systems are trained on tens of thousands of
hours of transcribed audio using large GPU clusters. Attempting to
"reinvent" that here would trade real mathematical depth (which the
attention model has, and which was verified via gradient checking) for a
shallow, undertrained result. Using an existing, well-established offline
engine for perception and generation, while keeping the actual
*understanding* (intent classification) as the from-scratch, verified
component, is the correct engineering scope decision.

**Language detection without a dedicated model.** Rather than training a
separate language-identification classifier, the same recorded audio is
passed through both the English and Arabic acoustic models, and whichever
one returns a longer, higher-confidence transcription is taken as the
detected language. This is a simple, explainable heuristic exploiting the
fact that each acoustic model is only well-calibrated for the sounds of
its own language.

**Honest limitation -- Egyptian Arabic ASR accuracy.** The offline Vosk
Arabic model is trained on Modern Standard Arabic (MSA), not Egyptian
Arabic specifically. Since Egyptian Arabic differs from MSA in
pronunciation and vocabulary, real-world recognition accuracy on Egyptian
speech will be noticeably lower than on English speech, or than a
cloud-based dialect-aware engine would achieve. This tradeoff was made
deliberately in favor of full offline/private operation (see the
project's design goals in the top-level README), and is stated here
rather than left to be discovered as a surprise. A natural extension
would be fine-tuning Vosk's acoustic model on Egyptian-dialect audio, or
swapping in a cloud API when internet access and lower privacy
requirements are both acceptable.

**Target normalization.** Since a user might name an app or file in
either language ("chrome" or "كروم"), execution requires mapping whatever
surface form was heard to the actual Windows command -- handled by the
reverse lookup table in `lexicon.py`, separate from the learned model.
This mirrors the same design principle from Section 6: the neural network
handles the part that benefits from learning (intent), while a simple,
auditable rule-based layer handles deterministic lookup.

## 10. Ablation Study: Does Attention Actually Use Word Order?

Section 5 found that Naive Bayes matches the attention model on the main
task, and explained that this is because the main task doesn't strongly
require word order -- intent is mostly determined by which words appear.
That explanation is only credible if it's actually tested, not just
asserted. This section reports that test.

**Design.** A separate, controlled diagnostic task (`synthetic_order_task.py`)
was constructed so that word order is the *only* usable signal: each
example is a 4-token sequence containing exactly one `MARK_X` and one
`MARK_Y` token (plus two label-independent filler tokens), with the label
defined as `1` if `MARK_X` precedes `MARK_Y`, else `0`.

Because every example contains exactly one of each marker, the
bag-of-words count vector for `MARK_X`/`MARK_Y` is the constant `(1, 1)`
across all examples regardless of label -- meaning the mutual information
between the bag-of-words representation and the label is provably zero:
$I(\text{bag-of-words}(x); y) = 0$. This gives a hard, mathematically
justified 50% accuracy ceiling for any order-blind model, rather than an
empirical "it happened to do worse" result that could be attributed to
undertraining or capacity limits.

**Models compared:**
1. The full attention model (attention + positional encoding) from
   Sections 2-3.
2. `OrderBlindClassifier` (`network.py`) -- the identical Embedding +
   Dense head, but with attention and positional encoding both removed:
   just a masked mean-pool directly over token embeddings. This isolates
   exactly the property being tested, rather than comparing against a
   different model family.
3. Naive Bayes, included again since it is also inherently a
   bag-of-words model, as a second independent check.

**Results** (`ablation_order_sensitivity.py`, plotted in
`ablation_order_sensitivity.png`):

| Model | Validation Accuracy |
|---|---|
| Full attention (order-aware) | 100.0% |
| Order-blind neural baseline | 53.3% |
| Naive Bayes | 50.7% |

Both order-blind models land within a few points of the predicted 50%
chance ceiling (the small deviation is expected finite-sample noise), while
the attention model solves the task completely. This is now direct
empirical confirmation of the permutation-invariance argument in Section
3.4 and the explanation offered in Section 5 -- not an assumption carried
over from the architecture's theoretical design, but a result measured on
a task built specifically to isolate that one claim.

## 11. Dataset Realism: Beyond Template Generation

The original bilingual dataset (Section 8) was entirely template-generated:
a fixed set of sentence patterns with the target word substituted in. This
is a real limitation -- template data cannot capture the irregularity of
genuine spontaneous speech (typos, contractions, sentence fragments,
code-switching, casual filler words), so accuracy on template-generated
validation data risks overstating how the model would perform on real
input.

**Fix.** 36 additional examples (20 English, 16 Egyptian Arabic) were
hand-authored one at a time rather than generated from templates --
including casual phrasing ("yo open chrome", "spotify pls"), sentence
fragments, and colloquial Egyptian constructions not captured by the
template grammar ("هوا فين النوت باد افتحه" -- roughly "where's the
notepad, open it"). Each example is tagged with its source
(`template` vs `hand_written`) so evaluation can report them separately
rather than blending them into one aggregate number.

**Result:** validation accuracy on the hand-written subset was 100%
(n=9), matching the 98.8% on template-generated validation examples
(n=83) -- the model's performance held up on the messier, non-templated
examples rather than being an artifact of template regularity. The small
hand-written validation sample size (9 examples) is itself an honest
limitation: this is encouraging, not conclusive, and a stronger version of
this fix would involve significantly more hand-authored or genuinely
crowd-sourced data.

## 12. Testing the Voice Pipeline

The voice components (`voice_io.py`, `voice_assistant.py`) could not be
executed end-to-end in the environment this project was built in -- there
is no microphone and no internet access available there to download the
~40-300MB Vosk model files. Rather than leaving that layer entirely
unverified, the one piece of logic in the voice pipeline that doesn't
depend on audio hardware -- deciding which language a transcription
belongs to (`choose_best_language()`) -- was extracted into a standalone,
pure function and covered with real unit tests
(`test_voice_language_detection.py`), including a regression case where a
high-confidence *empty* result should never be preferred over a
low-confidence *non-empty* one. All five tests pass.

This does not substitute for testing actual speech recognition accuracy
on real recorded audio, which requires the target machine's microphone and
the downloaded models -- that remains an open item for you to verify and
report back on, and the honest thing to do here is say so plainly rather
than imply the whole voice stack was validated when only its decision
logic was.

## 13. PyTorch Reimplementation

Everything above (Sections 2-12) uses a hand-derived, from-scratch numpy
implementation, verified via gradient checking. That verification was the
point: it demonstrates the mathematics is understood well enough to
implement and check correctly, not just used as a black box. Having
established that, there is no remaining pedagogical reason to avoid
standard libraries for further work -- reinventing autograd repeatedly
adds risk without adding insight once the underlying math has already been
derived and verified once.

`network_torch.py` reimplements the same architecture using PyTorch's
`nn.Module` and autograd, with two upgrades that are nearly free once a
real library is in use (both were flagged as future work in Section 7):

- **Multi-head attention** (4 heads instead of 1), via `nn.MultiheadAttention`
- **Adam optimizer** instead of plain SGD, via `torch.optim.Adam`

**Why this reduces error risk rather than just being "easier":** the
numpy version's correctness rests on a hand-derived backward pass that
had to be checked numerically to be trusted (Section 4). The PyTorch
version's correctness rests instead on `autograd`, a component tested
across millions of real deployments -- shifting risk away from
custom-written matrix calculus and onto infrastructure that is
categorically better-tested than any individually-verified derivation can
be, however carefully checked.

**Honest verification status.** This file could not be executed in the
sandbox this project was built in -- there is no network access there to
install `torch`. What *could* be verified without installing torch: the
sinusoidal positional encoding formula in `network_torch.py` was
translated into a standalone numpy computation and directly compared
against the verified `attention.positional_encoding()` output --
identical to floating-point precision (max difference `0.0`), confirming
that specific piece of the translation is exact. The rest of the
PyTorch model (the attention layer, the training loop, the full forward
pass) has not been executed or checked by the assistant, unlike every
other component in this project.

`train_torch.py` includes explicit instructions for a 30-second
sanity check you can run yourself: install torch, run the script, and
confirm the loss decreases smoothly and validation accuracy lands in a
similar range to the numpy version's reported numbers (Section 8). If it
does, the two implementations are behaving consistently, which is the
correct standard of evidence here, since the two versions were never
expected to produce identical numbers (different optimizer, single- vs
multi-head attention, different random initialization).

## 14. Training Stability Fix

Two consecutive training runs (documented in the project's chat history)
showed the same problem: loss decreasing overall but with a visible spike
mid-training (e.g. loss jumping from ~0.3 back up to ~1.4 around epoch
180 in one run) before eventually converging. This is a known failure
mode of fixed-learning-rate full-batch SGD: as the model approaches a
sharper region of the loss surface, a constant step size can overshoot
and temporarily land somewhere worse before recovering.

**Fix applied:** two standard techniques, both implemented from scratch
consistent with the rest of this project:

1. **Exponential learning-rate decay** (`train.py`):
   $\text{lr}_{\text{epoch}} = \text{lr}_0 \times 0.995^{\text{epoch}}$,
   taking the learning rate from 0.3 down to ~0.067 over 300 epochs, so
   later updates are naturally smaller and less likely to overshoot.
2. **Global gradient-norm clipping** (`core.clip_grad_norm`): computes the
   L2 norm of every gradient in the model treated as one flattened vector,
   and rescales all gradients in place if that norm exceeds a threshold
   (5.0), preserving direction but capping magnitude -- a direct guard
   against the exact "one bad step" failure mode observed.

**Honest result:** re-running training after this fix produced a smooth,
monotonically decreasing loss curve with no spike (see
`training_curves.png`, and the gradient-norm trace in
`gradient_diagnostics.png`). However, gradient clipping never actually
triggered in this run -- the maximum observed gradient norm was 4.34,
under the 5.0 threshold. This means the fix that mattered here was the
learning-rate decay, not the clipping; clipping remains in place as a
safety net for future runs (e.g. with a higher initial learning rate or a
larger dataset) rather than as the mechanism that fixed this particular
instability. Attributing the fix correctly, rather than crediting both
changes equally because both were made, is the same standard of honesty
applied throughout this report.

One further honest note: final validation accuracy on this particular
stabilized run (96.7%) was slightly lower than the previous unstabilized
run's eventual peak (100%, reached only after recovering from its own
spike). With a dataset this small, run-to-run variation of a few
percentage points is expected noise, not evidence that stability and peak
accuracy trade off against each other in general -- multiple runs would
be needed to say anything stronger than that, and this report does not
claim more than what was actually observed.

## References

Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
(Architectural basis for scaled dot-product attention and sinusoidal
positional encoding, re-derived and re-implemented from scratch for this
project.)

## 15. Bugs Found and Fixed

A dedicated bug-finding pass turned up two real, non-obvious bugs, both
now fixed and verified. Neither was caught by the gradient check, since
gradient checking only verifies that the backward pass is mathematically
consistent with the forward pass -- it says nothing about whether the
inputs to that forward pass (the data, the initialization) are correct.
This is an important distinction: "the gradient check passed" and "the
implementation has no bugs" are not the same claim, and this section is
the honest record of finding bugs the gradient check couldn't see.

**Bug 1 -- non-reproducible dataset generation.** `dataset.py` used
Python's built-in `hash()` to derive a per-surface-form random seed for
template selection. Python randomizes string `hash()` per-process by
default (a security feature since Python 3.3, `PYTHONHASHSEED`) --
confirmed directly: calling `hash("chrome")` in three separate Python
processes returned three different values. This meant every "reproducible"
run with a fixed `seed=42` was silently generating *different* training
data each time, which is a very plausible contributor to the run-to-run
training instability discussed in Section 14 -- some of what looked like
pure optimizer noise was actually the dataset itself changing underneath
it. **Fix:** replaced `hash(surface)` with a stable `hashlib.md5`-based
hash. Verified by generating the dataset in three separate process
invocations and comparing an MD5 checksum of the resulting text: now
identical across all three, where it previously would not have been.

**Bug 2 -- correlated layer initialization.** Every layer in
`TransformerIntentClassifier` (embedding, attention, both dense layers)
was initialized with the literal same `seed` value. Since each layer
constructs its own fresh `np.random.default_rng(seed)`, layers of
compatible shape drew from the *same* underlying random sequence --
confirmed directly: `dense1.W` and `dense2.W`, both seeded with 42,
shared identical leading values (`[0.088, -0.300, ...]`) before the fix.
This is a real initialization bug, not merely a style issue: correlated
initial weights across layers reduce the effective randomness the
optimizer has to break symmetry with, which is exactly why independent
initialization is standard practice. **Fix:** each layer now receives a
distinct derived seed (`seed`, `seed+1`, `seed+2`, ...). Verified directly
by re-inspecting the same weight slices post-fix (no longer matching) and
by re-running the full test suite (gradient check, main training, the
order-sensitivity ablation, and the voice-logic unit tests) to confirm no
regression -- all still pass.

**What this section demonstrates methodologically:** bug-hunting was
done by forming a specific, falsifiable hypothesis for each suspected bug
("if hash() is randomized, checksums across processes should differ"; "if
seeds are shared, these specific weight slices should be identical"),
then directly testing that hypothesis before writing any fix -- rather
than applying speculative changes and hoping they helped. Both hypotheses
were confirmed before the corresponding fix was written.

