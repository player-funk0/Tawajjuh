"""
voice_assistant.py -- Full voice loop:

    press Enter -> record mic -> transcribe (auto-detect English or
    Egyptian Arabic) -> classify intent with the from-scratch attention
    model -> execute the command -> speak a confirmation back in the
    same language that was detected.

Run this after:
  1. train.py has produced trained_model.npz
  2. Vosk models are downloaded into models/en and models/ar
     (see voice_io.py for exact setup steps)
  3. pip install vosk sounddevice pyttsx3
"""

from assistant import load_model, handle_command, execute
from voice_io import BilingualSTT, TTS

RESPONSES = {
    "en": {
        "opening_app": "Opening {target}",
        "opening_file": "Opening {target}",
        "low_confidence": "Sorry, I'm not sure what you meant.",
        "not_understood": "I didn't catch that, please try again.",
    },
    "ar": {
        "opening_app": "بفتح {target}",
        "opening_file": "بفتح {target}",
        "low_confidence": "معلش، مش فاهم قصدك.",
        "not_understood": "مسمعتش كويس، جرب تاني.",
    },
}


def main():
    print("Loading intent model...")
    model, vocab, intents, max_len = load_model()

    print("Loading speech models (this can take a few seconds)...")
    stt = BilingualSTT()
    tts = TTS()

    print("\nReady. Press Enter, then speak a command (English or Egyptian "
          "Arabic). Ctrl+C to quit.\n")

    while True:
        try:
            input("Press Enter to speak > ")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        audio = stt.record(seconds=4)
        text, lang, conf, candidates = stt.transcribe(audio)

        print(f"Heard [{lang}] (confidence {conf:.2f}): {text!r}")
        for other_lang, (other_text, other_conf) in candidates.items():
            if other_lang != lang:
                print(f"  (other model [{other_lang}] heard: {other_text!r}, conf {other_conf:.2f})")

        if not text.strip():
            msg = RESPONSES["en"]["not_understood"]
            print(msg)
            tts.speak(msg, lang="en")
            continue

        result = handle_command(text, model, vocab, intents, max_len)
        print(f"  -> intent: {result['intent']} ({result['confidence']*100:.1f}%) "
              f"target={result['canonical_target']}")

        responses = RESPONSES.get(lang, RESPONSES["en"])

        if result["confidence"] < 0.55:
            msg = responses["low_confidence"]
            print(msg)
            tts.speak(msg, lang=lang)
            continue

        exec_msg = execute(result["intent"], result["canonical_target"])
        print(exec_msg)

        key = "opening_app" if result["intent"] == "OPEN_APP" else "opening_file"
        spoken = responses[key].format(target=result["canonical_target"])
        tts.speak(spoken, lang=lang)


if __name__ == "__main__":
    main()
