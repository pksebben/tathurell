"""Smoke tests for the CHiME-6 corpus loader (eval/corpora/chime6.py).

These tests require only the transcription JSON (eval/data/chime6/), NOT
the multi-GB audio tarball. Audio-slicing is tested elsewhere if/when the
dev tarball is locally available.
"""
import glob

from eval.corpora.chime6 import load


def _s02_path() -> str:
    """Return the S02 dev-session JSON path, or raise a clear error."""
    matches = sorted(
        glob.glob("eval/data/chime6/**/S02*.json", recursive=True)
    )
    if not matches:
        raise FileNotFoundError(
            "CHiME-6 S02 transcription JSON not found under eval/data/chime6/. "
            "Download CHiME6_transcriptions.tar.gz from openslr.org/150 and "
            "extract it there."
        )
    return matches[0]


def test_chime6_loads_multispeaker_reference():
    """Full session Reference must have many segments and >= 3 speakers."""
    ref, audio = load(_s02_path())
    assert audio is None, "No audio_out was requested, expected None"
    assert len(ref.segments) > 10, (
        f"Expected > 10 segments, got {len(ref.segments)}"
    )
    speakers = {s.speaker for s in ref.segments}
    # Dinner-party sessions have 4 speakers; require at least 3 as a safe floor.
    assert len(speakers) >= 3, (
        f"Expected >= 3 speakers, got {len(speakers)}: {speakers}"
    )


def test_chime6_clip_to_max_seconds():
    """Clipping to max_seconds must exclude late utterances and clip end times."""
    window = 120.0  # 2-minute excerpt
    ref, audio = load(_s02_path(), max_seconds=window)
    assert audio is None, "No audio_out requested, expected None"
    assert len(ref.segments) > 0, "Expected segments in first 2 minutes"
    for seg in ref.segments:
        assert seg.start < window, (
            f"Segment start {seg.start:.2f}s >= max_seconds {window}"
        )
        assert seg.end <= window, (
            f"Segment end {seg.end:.2f}s > max_seconds {window}"
        )


def test_chime6_uri():
    """Reference URI must encode the session name."""
    ref, _ = load(_s02_path(), session="S02")
    assert "S02" in ref.uri, f"Expected 'S02' in URI, got {ref.uri!r}"
