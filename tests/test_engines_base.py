from eval.engines.base import assign_speakers_max_overlap


def test_word_assigned_to_max_overlap_turn(turns, words):
    out = assign_speakers_max_overlap(words, turns)
    assert [w["speaker"] for w in out] == ["A", "B", "A"]  # last word past turns -> nearest (A)


def test_overlap_picks_greater_overlap_not_first():
    turns = [
        {"speaker": "A", "start": 0.0, "end": 2.1},
        {"speaker": "B", "start": 2.0, "end": 5.0},
    ]
    # word [2.0,2.9]: 0.1s overlap with A, 0.9s with B -> B
    out = assign_speakers_max_overlap([{"word": "w", "start": 2.0, "end": 2.9}], turns)
    assert out[0]["speaker"] == "B"


def test_empty_turns_assigns_none():
    out = assign_speakers_max_overlap([{"word": "w", "start": 0.0, "end": 1.0}], [])
    assert out[0]["speaker"] is None
