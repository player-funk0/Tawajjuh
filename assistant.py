"""
assistant.py -- Loads the trained attention model (trained_model.npz) and
uses it to classify and execute real commands in English OR Egyptian
Arabic: "open notepad", "افتح الكروم", "show me my resume", etc.
"""

import os
import subprocess
import numpy as np

from dataset import Vocab, tokenize
from network import TransformerIntentClassifier
from lexicon import ALL_STOPWORDS, normalize_target, app_command


def extract_target(text):
    words = [w for w in tokenize(text) if w not in ALL_STOPWORDS]
    return words if words else tokenize(text)


def load_model(path="trained_model.npz"):
    ckpt = np.load(path, allow_pickle=True)
    vocab = Vocab.from_dict({"words": list(ckpt["vocab_words"])})
    intents = list(ckpt["intents"])
    max_len = int(ckpt["max_len"])

    model = TransformerIntentClassifier(
        vocab_size=len(vocab), max_len=max_len,
        d_model=int(ckpt["d_model"]), d_k=int(ckpt["d_k"]), hidden=int(ckpt["hidden"]),
        num_classes=len(intents),
    )
    model.embedding.table = ckpt["embedding_table"]
    model.attn.Wq, model.attn.Wk, model.attn.Wv = ckpt["Wq"], ckpt["Wk"], ckpt["Wv"]
    model.dense1.W, model.dense1.b = ckpt["dense1_W"], ckpt["dense1_b"]
    model.dense2.W, model.dense2.b = ckpt["dense2_W"], ckpt["dense2_b"]

    return model, vocab, intents, max_len


def predict(text, model, vocab, intents, max_len):
    ids, mask = vocab.encode(text, max_len)
    ids = np.array([ids])
    mask = np.array([mask])
    logits = model.forward(ids, mask)
    # softmax for interpretable probability, done manually (exp/sum), matching core.softmax
    z = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)
    idx = probs.argmax(axis=1)[0]
    return intents[idx], float(probs[0][idx])


def execute(intent, canonical_target):
    if intent == "OPEN_APP":
        command = app_command(canonical_target)
        subprocess.Popen(f"start {command}", shell=True)
        return f"Opening app: {canonical_target}"
    elif intent == "OPEN_FILE":
        try:
            os.startfile(canonical_target)
            return f"Opening file: {canonical_target}"
        except FileNotFoundError:
            return f"Couldn't find file '{canonical_target}' in the current folder."
    return f"Unknown intent: {intent}"


def handle_command(text, model, vocab, intents, max_len):
    """Runs the full pipeline for one command; returns a dict of results
    (used by both the text CLI below and voice_assistant.py)."""
    intent, confidence = predict(text, model, vocab, intents, max_len)
    target_words = extract_target(text)
    canonical_target = normalize_target(intent, target_words)
    return {
        "intent": intent,
        "confidence": confidence,
        "raw_target_words": target_words,
        "canonical_target": canonical_target,
    }


def main():
    model, vocab, intents, max_len = load_model()
    print("From-scratch attention-based intent classifier loaded (English + Egyptian Arabic).")
    print("Type a command like 'open notepad' or 'افتح الكروم', or 'quit' to exit.\n")

    while True:
        text = input("Command > ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue

        result = handle_command(text, model, vocab, intents, max_len)
        print(f"  -> intent: {result['intent']}  (confidence {result['confidence']*100:.1f}%)")
        print(f"  -> target: {result['canonical_target']}")

        if result["confidence"] < 0.55:
            print("  -> low confidence, skipping execution.\n")
            continue

        print(f"  -> {execute(result['intent'], result['canonical_target'])}\n")


if __name__ == "__main__":
    main()

