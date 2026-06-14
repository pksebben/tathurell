"""Smoke test for the WhisperX engine adapter.

The full transcribe() path requires HF_TOKEN (pyannote diarization model
download).  Without a token this test is skipped — that is the expected
current state and is not a failure.

To run the full test, export HF_TOKEN and execute:
    pytest tests/test_whisperx_engine_smoke.py -v
"""
import os
import pytest
from eval.engines.whisperx_stack import WhisperXStack


@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="needs HF_TOKEN (pyannote diarization)")
def test_whisperx_runs_on_clip():
    words = WhisperXStack().transcribe("dollop_test_a.mp3")
    assert len(words) > 50
    assert all("speaker" in w for w in words)
