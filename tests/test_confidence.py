from tathurell.confidence import word_confidences

# diar_segments: list of (start, end, speaker)
DIAR = [(0.0, 1.0, "A"), (1.0, 2.0, "B")]


def test_clean_word_is_confident():
    words = [{"start": 0.1, "end": 0.5, "speaker": "A"}]
    assert word_confidences(words, DIAR) == [1.0]


def test_boundary_word_is_split():
    # Word 0.5-1.5 overlaps A for 0.5 and B for 0.5; assigned to A -> 0.5.
    words = [{"start": 0.5, "end": 1.5, "speaker": "A"}]
    assert word_confidences(words, DIAR) == [0.5]


def test_word_assigned_to_speaker_with_no_overlap_is_zero():
    # Realign can move a word to a speaker that has no local diarization overlap.
    words = [{"start": 0.1, "end": 0.5, "speaker": "B"}]
    assert word_confidences(words, DIAR) == [0.0]


def test_gap_word_is_zero():
    # Word entirely outside any diarization segment (fill_nearest territory).
    words = [{"start": 5.0, "end": 5.5, "speaker": "A"}]
    assert word_confidences(words, DIAR) == [0.0]


def test_empty_diarization_is_all_zero():
    words = [{"start": 0.1, "end": 0.5, "speaker": "A"}]
    assert word_confidences(words, []) == [0.0]


def test_order_preserved_for_multiple_words():
    words = [
        {"start": 0.1, "end": 0.5, "speaker": "A"},   # 1.0
        {"start": 0.5, "end": 1.5, "speaker": "B"},   # overlaps A .5, B .5 -> B -> 0.5
    ]
    assert word_confidences(words, DIAR) == [1.0, 0.5]
