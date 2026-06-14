import os
import pytest
from eval.engines.mlxwhisper_pyannote import MlxWhisperPyannote


@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="needs HF_TOKEN (pyannote diarization)")
def test_mlx_runs_on_clip():
    words = MlxWhisperPyannote().transcribe("dollop_test_a.mp3")
    assert len(words) > 50
    assert all("speaker" in w for w in words)
