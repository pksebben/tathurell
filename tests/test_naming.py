from tathurell.naming import group_by_speaker, apply_names, render_runs


def test_group_starts_new_run_with_triggering_word():
    words = [
        {"word": "hello", "speaker": "A"},
        {"word": "there", "speaker": "A"},
        {"word": "hi", "speaker": "B"},
        {"word": "again", "speaker": "A"},
    ]
    assert group_by_speaker(words) == [
        {"speaker": "A", "text": "hello there", "start": 0.0, "end": 0.0, "confidence": 1.0},
        {"speaker": "B", "text": "hi", "start": 0.0, "end": 0.0, "confidence": 1.0},
        {"speaker": "A", "text": "again", "start": 0.0, "end": 0.0, "confidence": 1.0},
    ]


def test_group_empty_input():
    assert group_by_speaker([]) == []


def test_group_single_speaker_concatenates():
    words = [{"word": "a", "speaker": "A"}, {"word": "b", "speaker": "A"}]
    assert group_by_speaker(words) == [
        {"speaker": "A", "text": "a b", "start": 0.0, "end": 0.0, "confidence": 1.0}
    ]


def test_group_handles_none_speaker():
    # WhisperX can leave a word's speaker unassigned (None); it groups under the
    # None key and apply_names falls back to the string label "None".
    words = [
        {"word": "x", "speaker": None},
        {"word": "y", "speaker": "A"},
    ]
    groups = group_by_speaker(words)
    assert groups == [
        {"speaker": None, "text": "x", "start": 0.0, "end": 0.0, "confidence": 1.0},
        {"speaker": "A", "text": "y", "start": 0.0, "end": 0.0, "confidence": 1.0},
    ]
    assert apply_names(groups, {"A": "dave"}) == "None: x\ndave: y"


def test_apply_names_formats_and_falls_back_to_label():
    groups = [{"speaker": "A", "text": "hello"}, {"speaker": "B", "text": "hi"}]
    assert apply_names(groups, {"A": "dave"}) == "dave: hello\nB: hi"


def test_group_carries_spans_and_min_confidence():
    words = [
        {"word": "a", "speaker": "A", "start": 0.0, "end": 0.5, "confidence": 0.9},
        {"word": "b", "speaker": "A", "start": 0.5, "end": 1.0, "confidence": 0.4},
        {"word": "c", "speaker": "B", "start": 1.0, "end": 1.5, "confidence": 1.0},
    ]
    groups = group_by_speaker(words)
    assert groups[0] == {
        "speaker": "A", "text": "a b", "start": 0.0, "end": 1.0, "confidence": 0.4,
    }
    assert groups[1] == {
        "speaker": "B", "text": "c", "start": 1.0, "end": 1.5, "confidence": 1.0,
    }


def test_render_runs_merges_consecutive_same_speaker():
    runs = [
        {"speaker": "Alice", "text": "hello"},
        {"speaker": "Alice", "text": "there"},
        {"speaker": "Bob", "text": "hi"},
    ]
    assert render_runs(runs) == "Alice: hello there\nBob: hi"
