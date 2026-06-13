from eval.metrics.wer import word_error_rate


def test_identical_is_zero():
    assert word_error_rate("the quick brown fox", "the quick brown fox") == 0.0


def test_one_substitution_in_four_words():
    # normalization lowercases; one wrong word out of 4 -> 0.25
    assert word_error_rate("the quick brown fox", "the QUICK brown DOG") == 0.25
