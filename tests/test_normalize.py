from eval.metrics.normalize import normalize_text


def test_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, World!") == "hello world"


def test_idempotent():
    once = normalize_text("Mr. Smith paid $5.")
    assert normalize_text(once) == once
