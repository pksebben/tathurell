import pytest


@pytest.fixture
def turns():
    # diarization turns: (speaker, start, end), chronological
    return [
        {"speaker": "A", "start": 0.0, "end": 5.0},
        {"speaker": "B", "start": 5.0, "end": 10.0},
        {"speaker": "A", "start": 10.0, "end": 15.0},
    ]


@pytest.fixture
def words():
    # asr words with absolute start/end
    return [
        {"word": "hello", "start": 0.5, "end": 1.0},
        {"word": "there", "start": 6.0, "end": 6.4},   # inside B
        {"word": "bye", "start": 16.0, "end": 16.5},   # past last turn -> nearest A
    ]
