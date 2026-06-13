from eval.metrics.cpwer import cp_wer


def test_perfect_match_zero():
    ref = {"A": "hello world", "B": "good morning"}
    hyp = {"A": "hello world", "B": "good morning"}
    assert cp_wer(reference=ref, hypothesis=hyp) == 0.0


def test_speaker_swap_still_matched_by_cpwer():
    # cpWER finds the best speaker permutation, so swapped labels with correct
    # words still score 0.
    ref = {"A": "hello world", "B": "good morning"}
    hyp = {"X": "good morning", "Y": "hello world"}
    assert cp_wer(reference=ref, hypothesis=hyp) == 0.0
