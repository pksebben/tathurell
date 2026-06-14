"""Smoke tests for the AMI corpus loader (eval/corpora/ami.py).

These tests hit the HuggingFace hub on first run (streaming, ~meeting slice only).
"""
import tempfile
import os

from eval.corpora.ami import load, AMI_MEETING_ID, _SAMPLE_RATE

MAX_SECONDS_SMOKE = 60.0


def test_ami_loads_multispeaker_reference():
    ref, _ = load(condition="sdm")
    assert len(ref.segments) > 10
    assert len({s.speaker for s in ref.segments}) >= 2
    assert all(s.end >= s.start for s in ref.segments)


def test_ami_max_seconds_clips_reference():
    """load(max_seconds=60) should produce segments all ending at or before 60s.

    This tests the time-excerpting path added for tractability with CPU WhisperX.
    Assertion is intentionally light — we check the boundary constraint only,
    not exact segment counts (which vary by meeting).
    """
    ref, _ = load(condition="sdm", max_seconds=MAX_SECONDS_SMOKE)
    assert len(ref.segments) > 0, "Expected at least one segment in the first 60s"
    assert max(s.end for s in ref.segments) <= MAX_SECONDS_SMOKE, (
        f"max(segment.end) exceeds max_seconds={MAX_SECONDS_SMOKE}"
    )


def test_ami_audio_reconstruction_duration():
    """Audio written to disk should span roughly the length of the meeting.

    Checks that the meeting-aligned reconstruction places audio at the correct
    timeline position: written duration ≈ max(end_time) of segments.
    """
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        ref, out_path = load(condition="sdm", meeting_id=AMI_MEETING_ID, audio_out=tmp_path)
        assert out_path == tmp_path
        assert os.path.exists(tmp_path)

        info = sf.info(tmp_path)
        max_end = max(s.end for s in ref.segments)

        # Written duration should be within 1 second of the last reference end_time.
        assert abs(info.duration - max_end) < 1.0, (
            f"Audio duration {info.duration:.2f}s differs from max end_time "
            f"{max_end:.2f}s by more than 1s — reconstruction may be misaligned"
        )
        assert info.samplerate == _SAMPLE_RATE, (
            f"Expected {_SAMPLE_RATE} Hz, got {info.samplerate}"
        )
        print(
            f"\n[ami audio check] written duration={info.duration:.2f}s, "
            f"max end_time={max_end:.2f}s, delta={abs(info.duration - max_end):.3f}s"
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
