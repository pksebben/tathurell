import os
import pytest
from eval.engines.vosk_pyannote import VoskPyannote


@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="needs HF_TOKEN")
def test_vosk_engine_runs_on_clip():
    words = VoskPyannote().transcribe("dollop_test_a.mp3")
    assert len(words) > 50
    assert all("speaker" in w for w in words)
    assert {w["speaker"] for w in words} != {None}
