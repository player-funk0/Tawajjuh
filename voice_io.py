"""
voice_io.py -- Offline speech-to-text (Vosk) and text-to-speech (pyttsx3).

Fully offline: no internet connection needed after the one-time model
download. This matches the project's original design goal (a private,
local device-control agent). Be aware: offline Egyptian Arabic
recognition is noticeably weaker than a cloud engine would be, because
Vosk's Arabic model is trained on Modern Standard Arabic (MSA), not the
Egyptian dialect specifically -- see report.md for an honest discussion
of this tradeoff and what it means for real-world reliability.

===========================================================================
ONE-TIME SETUP
===========================================================================

1. Install packages:
       pip install vosk sounddevice pyttsx3

2. Download language models from https://alphacephei.com/vosk/models
   and unzip into a `models/` folder next to this file:

       models/en/   <- vosk-model-small-en-us-0.15   (~40 MB)
       models/ar/   <- vosk-model-ar-mgb2-0.4         (~300 MB)

   (Rename the unzipped folders to exactly "en" and "ar".)

3. For Arabic TEXT-TO-SPEECH output, Windows needs an Arabic voice
   installed: Settings > Time & Language > Speech > Manage voices >
   Add voices > Arabic. Without this, Arabic responses will be spoken
   with whatever default voice is installed (usually English), or
   silently skipped -- the code below handles this gracefully and warns
   you rather than crashing.

===========================================================================
HOW LANGUAGE DETECTION WORKS
===========================================================================

Rather than training a separate language-identification model (which
would need its own labeled audio dataset), this takes a simpler and
fully valid engineering approach: run the SAME recorded audio through
BOTH the English and Arabic acoustic models, and take whichever one
returns a more confident, non-empty transcription. Each model only
"understands" its own language's sounds well, so the correct model
tends to produce a higher word-confidence score. This isn't perfect
(short utterances make it noisier), but it's an honest, explainable
heuristic rather than a black box.
"""

import json
import numpy as np


class BilingualSTT:
    def __init__(self, en_model_path="models/en", ar_model_path="models/ar", sample_rate=16000):
        import sounddevice as sd  # imported here so the rest of the file
        from vosk import Model    # can still be read/reviewed even if these
        self._sd = sd             # packages aren't installed yet.
        self.sample_rate = sample_rate

        self.models = {}
        for lang, path in (("en", en_model_path), ("ar", ar_model_path)):
            try:
                self.models[lang] = Model(path)
            except Exception as e:
                print(f"Warning: could not load '{lang}' Vosk model from {path}: {e}")

        if not self.models:
            raise RuntimeError(
                "No Vosk models loaded. See the setup instructions at the "
                "top of voice_io.py -- you need to download and place the "
                "model folders before this will work."
            )
        print(f"Loaded speech models for: {list(self.models.keys())}")

    def record(self, seconds=4):
        """Records `seconds` of audio from the default microphone."""
        print(f"Listening... ({seconds}s)")
        audio = self._sd.rec(
            int(seconds * self.sample_rate),
            samplerate=self.sample_rate, channels=1, dtype="int16",
        )
        self._sd.wait()
        return audio.tobytes()

    def _transcribe_with_model(self, audio_bytes, model):
        from vosk import KaldiRecognizer
        rec = KaldiRecognizer(model, self.sample_rate)
        rec.SetWords(True)
        rec.AcceptWaveform(audio_bytes)
        result = json.loads(rec.FinalResult())
        text = result.get("text", "")
        words = result.get("result", [])
        avg_conf = float(np.mean([w.get("conf", 0.0) for w in words])) if words else 0.0
        return text, avg_conf

    def transcribe(self, audio_bytes):
        """
        Returns (text, detected_language, confidence, all_candidates).
        See module docstring and choose_best_language() for how language
        is chosen.
        """
        candidates = {}
        for lang, model in self.models.items():
            text, conf = self._transcribe_with_model(audio_bytes, model)
            candidates[lang] = (text, conf)

        best_lang = choose_best_language(candidates)
        text, conf = candidates[best_lang]
        return text, best_lang, conf, candidates


def choose_best_language(candidates):
    """
    Pure decision function, deliberately separated from BilingualSTT so it
    can be unit-tested without any microphone, audio, or Vosk model --
    see test_voice_language_detection.py. This was the one piece of the
    voice pipeline that could actually be verified in an environment with
    no audio hardware, so it's structured to make that possible rather
    than being buried inline inside a method that requires real audio to
    exercise.

    candidates: dict of lang -> (text, avg_word_confidence)
    Returns: the language key with the best candidate, preferring any
    non-empty transcription over an empty one, then higher confidence.
    """
    return max(candidates, key=lambda l: (len(candidates[l][0]) > 0, candidates[l][1]))


class TTS:
    """Offline text-to-speech using whatever voices are installed on Windows."""

    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty("voices")
        self.en_voice_id = self._find_voice(["english", "en_", "en-"])
        self.ar_voice_id = self._find_voice(["arabic", "ar_", "ar-"])

    def _find_voice(self, keywords):
        for v in self.voices:
            name = (v.name or "").lower()
            vid = (v.id or "").lower()
            if any(k in name or k in vid for k in keywords):
                return v.id
        return None

    def speak(self, text, lang="en"):
        voice_id = self.ar_voice_id if lang == "ar" else self.en_voice_id
        if voice_id:
            self.engine.setProperty("voice", voice_id)
        elif lang == "ar":
            print("(No Arabic voice installed on this system -- speaking "
                  "with the default voice instead. Add an Arabic voice via "
                  "Windows Settings > Time & Language > Speech.)")
        self.engine.say(text)
        self.engine.runAndWait()
