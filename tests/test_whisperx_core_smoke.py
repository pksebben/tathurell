import os
import pytest
from tathurell.whisperx_core import WhisperXTranscriber


@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="needs HF_TOKEN (pyannote)")
def test_transcriber_runs_on_clip():
    words = WhisperXTranscriber().transcribe("dollop_test_a.mp3")
    assert len(words) > 50
    assert all({"word", "start", "end", "speaker"} <= set(w) for w in words)
