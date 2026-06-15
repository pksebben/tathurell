from tathurell.realign import realign_speakers


def test_fixes_mid_sentence_sliver():
    # "These dudes are out of their minds." with 'of their' wrongly on SPEAKER_02.
    # Per-sentence majority (SPEAKER_01 owns 5/7 words) should reclaim the whole span.
    words = [
        {"word": "These", "speaker": "SPEAKER_01"},
        {"word": "dudes", "speaker": "SPEAKER_01"},
        {"word": "are", "speaker": "SPEAKER_01"},
        {"word": "out", "speaker": "SPEAKER_01"},
        {"word": "of", "speaker": "SPEAKER_02"},
        {"word": "their", "speaker": "SPEAKER_02"},
        {"word": "minds.", "speaker": "SPEAKER_01"},
    ]
    out = realign_speakers(words)
    assert [w["speaker"] for w in out] == ["SPEAKER_01"] * 7


def test_preserves_other_word_fields():
    words = [
        {"word": "out", "start": 0.0, "end": 0.1, "speaker": "SPEAKER_01"},
        {"word": "of", "start": 0.1, "end": 0.2, "speaker": "SPEAKER_02"},
        {"word": "minds.", "start": 0.2, "end": 0.4, "speaker": "SPEAKER_01"},
    ]
    out = realign_speakers(words)
    assert out[1]["start"] == 0.1 and out[1]["end"] == 0.2  # start/end carried through


def test_respects_sentence_boundary():
    # A genuine speaker change AT a sentence boundary must NOT be merged.
    words = [
        {"word": "Hello.", "speaker": "A"},
        {"word": "Hi", "speaker": "B"},
        {"word": "there.", "speaker": "B"},
    ]
    out = realign_speakers(words)
    assert [w["speaker"] for w in out] == ["A", "B", "B"]


def test_single_speaker_unchanged():
    words = [{"word": "a", "speaker": "A"}, {"word": "b.", "speaker": "A"}]
    out = realign_speakers(words)
    assert [w["speaker"] for w in out] == ["A", "A"]
