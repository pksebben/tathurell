from tathurell.naming import group_by_speaker, apply_names


def test_group_starts_new_run_with_triggering_word():
    words = [
        {"word": "hello", "speaker": "A"},
        {"word": "there", "speaker": "A"},
        {"word": "hi", "speaker": "B"},
        {"word": "again", "speaker": "A"},
    ]
    assert group_by_speaker(words) == [
        {"speaker": "A", "text": "hello there"},
        {"speaker": "B", "text": "hi"},
        {"speaker": "A", "text": "again"},
    ]


def test_group_empty_input():
    assert group_by_speaker([]) == []


def test_group_single_speaker_concatenates():
    words = [{"word": "a", "speaker": "A"}, {"word": "b", "speaker": "A"}]
    assert group_by_speaker(words) == [{"speaker": "A", "text": "a b"}]


def test_group_handles_none_speaker():
    # WhisperX can leave a word's speaker unassigned (None); it groups under the
    # None key and apply_names falls back to the string label "None".
    words = [
        {"word": "x", "speaker": None},
        {"word": "y", "speaker": "A"},
    ]
    groups = group_by_speaker(words)
    assert groups == [{"speaker": None, "text": "x"}, {"speaker": "A", "text": "y"}]
    assert apply_names(groups, {"A": "dave"}) == "None: x\ndave: y"


def test_apply_names_formats_and_falls_back_to_label():
    groups = [{"speaker": "A", "text": "hello"}, {"speaker": "B", "text": "hi"}]
    assert apply_names(groups, {"A": "dave"}) == "dave: hello\nB: hi"
