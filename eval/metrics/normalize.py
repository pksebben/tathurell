"""Single shared normalizer applied to EVERY engine output AND every reference
before any WER/cpWER computation, so engines that differ only in casing/
punctuation/number formatting (vosk: lowercase no-punct vs Whisper: cased,
punctuated) are compared fairly. Uses Whisper's EnglishTextNormalizer.
"""
from whisper.normalizers import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    return _normalizer(text)
