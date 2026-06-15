from tathurell.sampling import pick_speaker_samples


def test_picks_longest_run_not_first():
    words = [
        {"word": "hi", "start": 0.0, "end": 0.5, "speaker": "A"},
        {"word": "yo", "start": 0.5, "end": 1.0, "speaker": "B"},
        {"word": "this", "start": 1.0, "end": 1.4, "speaker": "A"},
        {"word": "is", "start": 1.4, "end": 1.8, "speaker": "A"},
        {"word": "longer", "start": 1.8, "end": 3.0, "speaker": "A"},
    ]
    out = pick_speaker_samples(words)
    assert out["A"]["start"] == 1.0
    assert out["A"]["text"] == "this is longer"
    assert out["B"]["text"] == "yo"


def test_caps_to_max_seconds():
    words = [
        {"word": "a", "start": 0.0, "end": 1.0, "speaker": "A"},
        {"word": "b", "start": 1.0, "end": 2.0, "speaker": "A"},
        {"word": "c", "start": 2.0, "end": 30.0, "speaker": "A"},
    ]
    out = pick_speaker_samples(words, max_seconds=8.0)
    assert out["A"]["start"] == 0.0
    assert out["A"]["end"] == 8.0
    assert out["A"]["text"] == "a b c"   # all three start < 8.0


def test_ignores_none_speaker_words():
    words = [
        {"word": "x", "start": 0.0, "end": 0.2, "speaker": None},
        {"word": "y", "start": 0.2, "end": 1.0, "speaker": "A"},
    ]
    out = pick_speaker_samples(words)
    assert set(out) == {"A"}
    assert out["A"]["text"] == "y"
