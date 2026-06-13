"""Concatenated minimum-permutation WER (cpWER) via meeteval — the headline
metric. Scores per-speaker text against the reference under the best speaker
assignment, so it measures ASR + speaker attribution together. Both sides are
normalized first.
"""
from meeteval.wer.wer.cp import cp_word_error_rate
from eval.metrics.normalize import normalize_text


def cp_wer(reference: dict, hypothesis: dict) -> float:
    """reference/hypothesis: {speaker: concatenated text}."""
    ref = {spk: normalize_text(t) for spk, t in reference.items()}
    # meeteval accepts the hypothesis as a list of per-speaker strings.
    hyp = [normalize_text(t) for t in hypothesis.values()]
    return cp_word_error_rate(reference=ref, hypothesis=hyp).error_rate
