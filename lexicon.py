"""
lexicon.py -- Bilingual (English + Egyptian Arabic) vocabulary mapping.

A spoken command can name an app/file in English OR Egyptian Arabic
("open chrome" / "افتح الكروم"). The neural network only needs to learn
the INTENT (OPEN_APP vs OPEN_FILE) -- but actually executing the command
requires mapping whatever surface word was spoken to a real Windows
command. This file is that mapping table, in both directions:

  canonical name -> list of surface forms (English + Egyptian Arabic)
                     used to build training data
  surface form  -> canonical name                (reverse lookup, used at
                                                    inference time to
                                                    normalize what was
                                                    heard into something
                                                    executable)

Note on Egyptian Arabic: the definite article "ال" (al-) attaches directly
to the noun with no space ("كروم" -> "الكروم", "chrome" -> "the chrome").
Rather than implementing Arabic morphological stripping (a real NLP
subfield on its own), both forms are listed explicitly as separate surface
forms. This is a deliberate scope decision, not an oversight.
"""

import re


def normalize_arabic(text):
    """
    Normalizes common Arabic spelling variation so informal/inconsistent
    spelling still matches the lexicon.

    BUG FIXED HERE: Egyptian Arabic is frequently typed/spoken without
    the hamza diacritic on alef -- "الإكسل" (with hamza) and "الاكسل"
    (without) are both extremely common for the same word, but an exact
    string match treats them as entirely different tokens. This silently
    broke target normalization: the intent classifier correctly predicted
    OPEN_APP for "عايز الاكسل دلوقتي", but the target lookup failed to
    map "الاكسل" to the canonical "excel" (only "الإكسل" was in the
    lexicon), so execution would have tried to run a literally nonexistent
    Windows command. This normalization is applied consistently both when
    building the lexicon's reverse lookup table and when normalizing a
    target at inference time, so the two sides can't drift apart again.

    Normalizations applied (standard for Arabic NLP preprocessing):
      - Alef variants (أ, إ, آ) -> bare alef (ا)
      - Ta marbuta (ة) -> ha (ه)
      - Alef maksura (ى) -> ya (ي)
    """
    text = re.sub("[أإآ]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    return text

# canonical_target -> (windows_command, [surface forms across both languages])
APP_LEXICON = {
    "notepad":    ("notepad",       ["notepad", "نوت باد", "النوت باد", "نوتباد"]),
    "chrome":     ("chrome",        ["chrome", "كروم", "الكروم", "جوجل كروم"]),
    "calculator": ("calc",          ["calculator", "الحاسبة", "حاسبة", "الآلة الحاسبة"]),
    "explorer":   ("explorer",      ["explorer", "الإكسبلورر", "مستكشف الملفات", "الملفات"]),
    "word":       ("winword",       ["word", "وورد", "الوورد"]),
    "excel":      ("excel",         ["excel", "إكسل", "الإكسل"]),
    "spotify":    ("spotify",       ["spotify", "سبوتيفاي", "السبوتيفاي"]),
    "vlc":        ("vlc",           ["vlc", "الفي إل سي", "في إل سي"]),
    "paint":      ("mspaint",       ["paint", "الرسام", "برنامج الرسم"]),
    "cmd":        ("cmd",           ["cmd", "الكوماند", "موجه الأوامر"]),
    "settings":   ("ms-settings:",  ["settings", "الإعدادات", "اعدادات"]),
    "terminal":   ("wt",            ["terminal", "التيرمينال", "الترمينال"]),
}

# canonical_target -> [surface forms]. Files don't have a "command" --
# they need a real filename+extension on disk to actually open, so the
# canonical name here is just the base name a user would refer to.
FILE_LEXICON = {
    "report":       ["report", "تقرير", "التقرير"],
    "notes":        ["notes", "ملاحظات", "الملاحظات"],
    "budget":       ["budget", "ميزانية", "الميزانية"],
    "photo":        ["photo", "صورة", "الصورة"],
    "presentation": ["presentation", "عرض تقديمي", "العرض التقديمي", "بوربوينت"],
    "resume":       ["resume", "سيرة ذاتية", "السيرة الذاتية"],
    "song":         ["song", "أغنية", "الأغنية"],
    "video":        ["video", "فيديو", "الفيديو"],
    "todo":         ["todo", "مهام", "المهام"],
    "project":      ["project", "مشروع", "المشروع"],
}


def is_arabic(text):
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _build_reverse_map(lexicon, has_command):
    reverse = {}
    for canonical, value in lexicon.items():
        surface_forms = value[1] if has_command else value
        for s in surface_forms:
            reverse[normalize_arabic(s.lower())] = canonical
    return reverse


APP_SURFACE_TO_CANONICAL = _build_reverse_map(APP_LEXICON, has_command=True)
FILE_SURFACE_TO_CANONICAL = _build_reverse_map(FILE_LEXICON, has_command=False)


def app_command(canonical_name):
    entry = APP_LEXICON.get(canonical_name)
    return entry[0] if entry else canonical_name


def normalize_target(intent, raw_target_words):
    """
    raw_target_words: list of tokens extracted from the spoken/typed command
    (after stopword removal). Tries to match them (individually or joined)
    against the known surface forms for the given intent, returning the
    canonical name if found, or the raw joined text as a fallback.
    """
    lexicon_map = APP_SURFACE_TO_CANONICAL if intent == "OPEN_APP" else FILE_SURFACE_TO_CANONICAL
    normalized_words = [normalize_arabic(w) for w in raw_target_words]
    joined = " ".join(normalized_words)

    if joined in lexicon_map:
        return lexicon_map[joined]
    for w in normalized_words:
        if w in lexicon_map:
            return lexicon_map[w]
    # try matching multi-word Arabic phrases (e.g. "عرض تقديمي") against
    # any contiguous pair of tokens
    for i in range(len(normalized_words) - 1):
        pair = normalized_words[i] + " " + normalized_words[i + 1]
        if pair in lexicon_map:
            return lexicon_map[pair]

    return " ".join(raw_target_words)  # fallback: best-effort raw (unnormalized) text


# Stopwords in both languages, used to strip command/filler words before
# target extraction. Arabic forms include the "ال" (al-) prefixed variants
# where relevant since they appear as literal, distinct tokens.
STOPWORDS_EN = {
    "open", "launch", "start", "please", "can", "you", "run", "the",
    "app", "application", "file", "called", "for", "me", "up", "i",
    "want", "to", "show", "need", "see", "document", "a", "an", "of",
    "is", "it", "and", "go", "ahead", "would", "find", "pull",
}

STOPWORDS_AR = {
    "افتح", "شغل", "ممكن", "تفتح", "عايز", "افتحه", "لي", "لو", "سمحت",
    "يلا", "ابدأ", "ملف", "الملف", "بتاع", "وريني", "ورّيني", "اشوف",
    "هات", "من", "فضلك",
}

ALL_STOPWORDS = STOPWORDS_EN | STOPWORDS_AR
