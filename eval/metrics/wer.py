"""Speaker-agnostic WER via jiwer, on normalized text. Reference first."""
import jiwer
from eval.metrics.normalize import normalize_text


def word_error_rate(reference: str, hypothesis: str) -> float:
    return jiwer.wer(normalize_text(reference), normalize_text(hypothesis))
