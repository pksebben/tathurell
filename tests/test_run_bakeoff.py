"""Tests for eval/run_bakeoff.py.

Pure helper function tests (no token, no models) always run.
The integration smoke test is skipped without HF_TOKEN.
"""
from __future__ import annotations

import os

import pytest

from eval.run_bakeoff import words_to_hyp, words_to_rttm


# ---------------------------------------------------------------------------
# Pure helper: words_to_hyp
# ---------------------------------------------------------------------------

def test_words_to_hyp_two_speakers():
    """words_to_hyp groups words by speaker and builds correct flat text."""
    words = [
        {"word": "hello", "start": 0.0, "end": 0.5, "speaker": "A"},
        {"word": "world", "start": 0.5, "end": 1.0, "speaker": "A"},
        {"word": "hi",    "start": 1.0, "end": 1.3, "speaker": "B"},
        {"word": "there", "start": 1.3, "end": 1.8, "speaker": "B"},
    ]
    per_spk, flat = words_to_hyp(words)

    assert per_spk == {"A": "hello world", "B": "hi there"}, (
        f"Unexpected per_spk dict: {per_spk}"
    )
    assert flat == "hello world hi there", (
        f"Unexpected flat text: {flat!r}"
    )


def test_words_to_hyp_single_speaker():
    """Single-speaker input produces one entry and identical flat text."""
    words = [
        {"word": "one",   "start": 0.0, "end": 0.3, "speaker": "X"},
        {"word": "two",   "start": 0.3, "end": 0.6, "speaker": "X"},
        {"word": "three", "start": 0.6, "end": 0.9, "speaker": "X"},
    ]
    per_spk, flat = words_to_hyp(words)
    assert set(per_spk.keys()) == {"X"}
    assert per_spk["X"] == "one two three"
    assert flat == "one two three"


def test_words_to_hyp_empty():
    """Empty word list returns empty dict and empty flat string."""
    per_spk, flat = words_to_hyp([])
    assert per_spk == {}
    assert flat == ""


# ---------------------------------------------------------------------------
# Pure helper: words_to_rttm
# ---------------------------------------------------------------------------

def test_words_to_rttm_three_turns():
    """Speaker sequence A->B->A produces exactly 3 RTTM SPEAKER lines."""
    words = [
        {"word": "one",   "start": 0.0, "end": 0.5, "speaker": "A"},
        {"word": "two",   "start": 0.5, "end": 1.0, "speaker": "A"},
        {"word": "three", "start": 1.0, "end": 1.5, "speaker": "B"},
        {"word": "four",  "start": 1.5, "end": 2.0, "speaker": "B"},
        {"word": "five",  "start": 2.0, "end": 2.5, "speaker": "A"},
    ]
    uri = "test_uri"
    rttm = words_to_rttm(words, uri)

    lines = [l for l in rttm.splitlines() if l.strip()]
    assert len(lines) == 3, f"Expected 3 SPEAKER lines, got {len(lines)}:\n{rttm}"

    # Verify all lines are SPEAKER lines with the correct URI.
    for line in lines:
        assert line.startswith(f"SPEAKER {uri} 1 "), (
            f"Malformed RTTM line: {line!r}"
        )

    # Parse fields: SPEAKER <uri> <channel> <start> <dur> <NA> <NA> <spkr> <NA> <NA>
    def parse_line(l: str) -> tuple[float, float, str]:
        parts = l.split()
        return float(parts[3]), float(parts[4]), parts[7]

    turn0_start, turn0_dur, turn0_spk = parse_line(lines[0])
    turn1_start, turn1_dur, turn1_spk = parse_line(lines[1])
    turn2_start, turn2_dur, turn2_spk = parse_line(lines[2])

    # Turn 0: speaker A, starts at 0.0, duration = end of "two" - 0.0 = 1.0s
    assert turn0_spk == "A"
    assert abs(turn0_start - 0.0) < 1e-6, f"Turn 0 start: {turn0_start}"
    assert abs(turn0_dur - 1.0) < 1e-6, f"Turn 0 dur: {turn0_dur}"

    # Turn 1: speaker B, starts at 1.0, duration = 2.0 - 1.0 = 1.0s
    assert turn1_spk == "B"
    assert abs(turn1_start - 1.0) < 1e-6, f"Turn 1 start: {turn1_start}"
    assert abs(turn1_dur - 1.0) < 1e-6, f"Turn 1 dur: {turn1_dur}"

    # Turn 2: speaker A again, starts at 2.0, duration = 2.5 - 2.0 = 0.5s
    assert turn2_spk == "A"
    assert abs(turn2_start - 2.0) < 1e-6, f"Turn 2 start: {turn2_start}"
    assert abs(turn2_dur - 0.5) < 1e-6, f"Turn 2 dur: {turn2_dur}"


def test_words_to_rttm_empty():
    """Empty word list produces a single trailing newline (no SPEAKER lines)."""
    rttm = words_to_rttm([], "some_uri")
    lines = [l for l in rttm.splitlines() if l.strip()]
    assert lines == [], f"Expected no SPEAKER lines for empty input, got: {lines}"


def test_words_to_rttm_consecutive_same_speaker_merged():
    """All words from one speaker collapse into a single RTTM line."""
    words = [
        {"word": "a", "start": 0.0, "end": 0.2, "speaker": "S1"},
        {"word": "b", "start": 0.2, "end": 0.4, "speaker": "S1"},
        {"word": "c", "start": 0.4, "end": 0.6, "speaker": "S1"},
    ]
    rttm = words_to_rttm(words, "u")
    lines = [l for l in rttm.splitlines() if l.strip()]
    assert len(lines) == 1, f"Expected 1 line, got {len(lines)}: {rttm}"
    parts = lines[0].split()
    assert abs(float(parts[3]) - 0.0) < 1e-6  # start
    assert abs(float(parts[4]) - 0.6) < 1e-6  # duration


# ---------------------------------------------------------------------------
# Integration smoke (token-gated — skipped without HF_TOKEN)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    "HF_TOKEN" not in os.environ,
    reason="needs HF_TOKEN + pre-downloaded models",
)
def test_bakeoff_runs():
    """Full matrix smoke: run() completes and returns a non-empty row list."""
    import eval.run_bakeoff as rb
    rows = rb.run()
    assert rows, "run() returned an empty row list"
