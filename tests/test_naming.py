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


def test_apply_names_formats_and_falls_back_to_label():
    groups = [{"speaker": "A", "text": "hello"}, {"speaker": "B", "text": "hi"}]
    assert apply_names(groups, {"A": "dave"}) == "dave: hello\nB: hi"
