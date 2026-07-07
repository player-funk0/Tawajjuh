"""
test_voice_language_detection.py -- Actual, runnable unit tests for the
language-selection logic in voice_io.choose_best_language().

This is the one part of the voice pipeline that doesn't require a
microphone, an internet connection, or the ~300MB Vosk Arabic model to
verify -- so unlike the rest of voice_io.py / voice_assistant.py (which
are genuinely untested in this build environment, see README.md), this
file IS executed as part of this project, with real assertions and a
pass/fail result.

Run with: python test_voice_language_detection.py
"""

from voice_io import choose_best_language


def test_prefers_nonempty_over_empty():
    candidates = {
        "en": ("", 0.9),          # empty text despite high confidence
        "ar": ("افتح الكروم", 0.4),  # non-empty, lower confidence
    }
    assert choose_best_language(candidates) == "ar", \
        "Should prefer a non-empty transcription even with lower confidence"


def test_prefers_higher_confidence_when_both_nonempty():
    candidates = {
        "en": ("open chrome", 0.95),
        "ar": ("افتح شيء", 0.3),
    }
    assert choose_best_language(candidates) == "en", \
        "Should prefer the higher-confidence transcription when both are non-empty"


def test_both_empty_still_returns_a_language():
    candidates = {
        "en": ("", 0.1),
        "ar": ("", 0.05),
    }
    result = choose_best_language(candidates)
    assert result in ("en", "ar"), "Should still return a valid language key, not crash"


def test_single_language_available():
    candidates = {"en": ("open notepad", 0.8)}
    assert choose_best_language(candidates) == "en"


def test_low_confidence_nonempty_beats_high_confidence_empty():
    # Regression case: this is the actual scenario that motivated the
    # (has_text, confidence) tuple ordering rather than confidence alone --
    # Vosk can return a confident-looking score for an empty final result
    # in some silence/noise cases, which should never win over any real
    # transcription.
    candidates = {
        "en": ("", 0.99),
        "ar": ("م", 0.01),
    }
    assert choose_best_language(candidates) == "ar"


if __name__ == "__main__":
    tests = [
        test_prefers_nonempty_over_empty,
        test_prefers_higher_confidence_when_both_nonempty,
        test_both_empty_still_returns_a_language,
        test_single_language_available,
        test_low_confidence_nonempty_beats_high_confidence_empty,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -- {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
