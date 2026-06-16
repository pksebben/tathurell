import inspect
import os

import pytest

from tathurell.whisperx_core import WhisperXTranscriber


def test_transcribe_accepts_progress_param():
    # Lock the interface without needing models: progress is an optional kwarg.
    sig = inspect.signature(WhisperXTranscriber.transcribe)
    assert "progress" in sig.parameters
    assert sig.parameters["progress"].default is None


@pytest.mark.skipif(
    not os.environ.get("TATHURELL_E2E"),
    reason="slow real-model run (~2-3 min); set TATHURELL_E2E=1 to enable. "
    "Runs offline from the local HF cache -- no token needed.",
)
def test_transcriber_runs_on_clip_and_reports_progress():
    stages = []
    words = WhisperXTranscriber().transcribe("dollop_test_a.mp3", progress=stages.append)
    assert len(words) > 50
    assert all({"word", "start", "end", "speaker"} <= set(w) for w in words)
    # Coarse stages fire in order (each at most once, monotonic through the pipeline).
    assert stages == ["transcribing", "aligning", "diarizing", "finishing"]
    # Every word carries a diarization confidence in [0, 1].
    assert all(0.0 <= w["confidence"] <= 1.0 for w in words)
