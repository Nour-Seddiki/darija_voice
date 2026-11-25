"""Text normalisation for Algerian Darija (Arabic script + French/English code-switching).

Two levels:
  * normalize_target(): applied to training transcripts. Removes things that make targets
    inconsistent across sources (diacritics, <laugh>/<pause> tags, punctuation, Persian/Maghrebi
    letter variants) but keeps natural spelling.
  * normalize_for_wer(): stricter, applied to BOTH reference and hypothesis before scoring.
    Also folds hamza/alef variants, ta marbuta and alef maqsura, which dialect writers use
    interchangeably, so WER measures recognition, not spelling convention.
"""
import re
import unicodedata

# Arabic harakat, tanween, shadda, sukun, superscript alef, Quranic annotation marks, tatweel.
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
# <laugh>, <pause>, [music], (inaudible) ... style tags
_TAGS = re.compile(r"<[^<>]{0,40}>|\[[^\[\]]{0,40}\]")
# Letter variants -> the closest standard Arabic letter (consistent targets across writers).
_LETTER_MAP = str.maketrans({
    "ڨ": "ق",  # ڨ (Algerian /g/)  -> ق
    "ڤ": "ف",  # ڤ (/v/)            -> ف
    "چ": "ج",  # چ                  -> ج
    "ݘ": "ج",  # ݘ                  -> ج
    "پ": "ب",  # پ                  -> ب
    "ۉ": "و",  # ۉ                  -> و
    "ۋ": "و",  # ۋ                  -> و
    "ک": "ك",  # ک                  -> ك
    "گ": "ك",  # گ                  -> ك
    "ی": "ي",  # ی                  -> ي
    "ھ": "ه",  # ھ                  -> ه
    "ہ": "ه",  # ہ                  -> ه
    "ٱ": "ا",  # ٱ                  -> ا
    "ﷲ": "الله",  # ﷲ ligature -> الله
    # Arabic-Indic and Persian digits -> ASCII digits
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
})
# Anything that is not an Arabic letter, a Latin letter (incl. French accents), a digit,
# an apostrophe inside a word, or whitespace becomes a space.
_KEEP = re.compile(r"[^ء-غف-يa-zA-ZÀ-ɏ0-9' ]+")
_SPACES = re.compile(r"\s+")


def normalize_target(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = _TAGS.sub(" ", t)
    t = t.translate(_LETTER_MAP)
    t = _DIACRITICS.sub("", t)
    t = t.replace("’", "'").replace("`", "'")
    t = _KEEP.sub(" ", t)
    t = re.sub(r"(?<![a-zA-ZÀ-ɏ])'|'(?![a-zA-ZÀ-ɏ])", " ", t)  # stray quotes
    t = _SPACES.sub(" ", t).strip()
    return t.lower()  # only affects Latin letters


def normalize_display(text: str) -> str:
    """Readable training target for the speech LLM: same cleanup as normalize_target but keeps
    punctuation and case, so the model's written Darija stays natural."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    t = _TAGS.sub(" ", t)
    t = t.translate(_LETTER_MAP)
    t = _DIACRITICS.sub("", t)
    return _SPACES.sub(" ", t).strip()


_ALEF = re.compile(r"[آأإ]")  # آ أ إ -> ا


def normalize_for_wer(text: str) -> str:
    t = normalize_target(text)
    t = _ALEF.sub("ا", t)
    t = t.replace("ى", "ي")   # ى -> ي
    t = t.replace("ة", "ه")   # ة -> ه
    t = t.replace("ؤ", "و")   # ؤ -> و
    t = t.replace("ئ", "ي")   # ئ -> ي
    t = t.replace("'", "")
    return _SPACES.sub(" ", t).strip()


def arabic_char_count(text: str) -> int:
    return sum(1 for c in text if "ء" <= c <= "ي" or c.isalpha())


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    samples = [
        "السَّلامْ يَا النَّاسْ! واشْ رَاكُمْ؟ لابَاسْ؟ <laugh> أهَا!",
        "ڨُلت للمدارس ثاني، وَفّرولي غير ڨِسم نْقرّي فيه",
        "بزاف فايس تاع الحياة اليومية, mais surtout على لي فايس... c'est rien",
        "عندي ٣ دراري و ۲ بنات",
    ]
    for s in samples:
        print(repr(normalize_target(s)), "|", repr(normalize_for_wer(s)))
