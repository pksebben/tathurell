"""Smoke tests for the Earnings-21 corpus loader (eval/corpora/earnings21.py).

These tests fetch data from HuggingFace Hub and GitHub on first run.

Timing note: Earnings-21 NLP token files carry no per-segment timestamps.
All Segment start/end values are set to (0.0, audio_length) — DER is NOT
computable from this Reference. WER and cpWER are fully supported.
"""
import os
import tempfile

from eval.corpora.earnings21 import EARNINGS_CALL_ID, load


def test_earnings21_loads_multispeaker_reference():
    """Reference must have many segments and at least 2 distinct speakers."""
    ref, _ = load()
    assert len(ref.segments) > 10, (
        f"Expected > 10 segments, got {len(ref.segments)}"
    )
    speakers = {s.speaker for s in ref.segments}
    assert len(speakers) >= 2, (
        f"Expected >= 2 speakers, got {len(speakers)}: {speakers}"
    )


def test_earnings21_audio_write():
    """Audio written to disk should be non-empty and have a positive duration.

    Per-segment timing is unavailable (start=end=0.0 in Reference), so we
    cannot assert audio duration ≈ max(end_time). Instead we verify that the
    audio file is written, is non-zero length, and that soundfile can read it.
    """
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        ref, out_path = load(call_id=EARNINGS_CALL_ID, audio_out=tmp_path)
        assert out_path == tmp_path
        assert os.path.exists(tmp_path), "WAV file was not created"
        assert os.path.getsize(tmp_path) > 0, "WAV file is empty"

        info = sf.info(tmp_path)
        assert info.duration > 60.0, (
            f"Call audio is suspiciously short: {info.duration:.1f}s"
        )
        assert info.samplerate > 0, "Sample rate must be positive"

        print(
            f"\n[earnings21 audio check] duration={info.duration:.1f}s, "
            f"sr={info.samplerate}, speakers={len({s.speaker for s in ref.segments})}"
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
