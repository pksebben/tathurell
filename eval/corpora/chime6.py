"""CHiME-6 loader -> Reference from the per-session transcription JSON.

Dinner-party far-field recording corpus. Sessions are 2+ hours and contain
4 speakers (P05-P08 in S02, etc.). The transcription JSON is tiny (~2.4 MB
for all sessions); audio is only available as a monolithic 10 GB dev tarball
(CHiME6_dev.tar.gz from openslr.org/150) — supply a local audio path.

Audio note: CHiME-6 sessions are 2+ hours, far too long for CPU inference.
Use max_seconds to clip to a tractable excerpt. When audio_in and audio_out
are supplied, the excerpt wav is sliced from the full session audio so that
the audio timeline matches the clipped Reference segment times (both share
the same zero origin = session start).

JSON schema (confirmed 2026-06-13, openslr.org/150 CHiME6_transcriptions.tar.gz):
  session_id, speaker, start_time, end_time, words, ref, location
  Times are "HH:MM:SS.ss" strings.
"""
from __future__ import annotations

import json

from eval.corpora.base import Reference, Segment

KEY_SPEAKER = "speaker"
KEY_START = "start_time"
KEY_END = "end_time"
KEY_TEXT = "words"


def _t(v: str | float | int) -> float:
    """Parse a CHiME-6 time value to seconds.

    Handles both the "HH:MM:SS.ss" string format used in the JSON and plain
    numeric values (float/int) so the function is safe if the schema ever
    changes to numeric times.
    """
    if isinstance(v, (int, float)):
        return float(v)
    parts = v.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(v)


def load(
    json_path: str,
    session: str = "S02",
    max_seconds: float | None = None,
    audio_in: str | None = None,
    audio_out: str | None = None,
) -> tuple[Reference, str | None]:
    """Load a CHiME-6 session from its transcription JSON.

    Args:
        json_path:   Path to the per-session JSON file (e.g. dev/S02.json).
        session:     Session identifier string used to label the Reference URI.
        max_seconds: If set, keep only utterances whose start_time < max_seconds
                     and clip their end to max_seconds. Use this to produce a
                     tractable excerpt — CHiME-6 sessions are 2+ hours long.
        audio_in:    Path to the full-session distant-array wav file. Required
                     when audio_out is set.
        audio_out:   If given (and audio_in is given), write the clipped wav
                     excerpt [0, max_seconds] to this path.

    Returns:
        (Reference, audio_out_path_or_None) — audio_out is None when no audio
        slice was written, matching the contract of the AMI and Earnings-21
        loaders.
    """
    data = json.load(open(json_path))

    segments: list[Segment] = []
    for u in data:
        start = _t(u[KEY_START])
        end = _t(u[KEY_END])
        if max_seconds is not None:
            if start >= max_seconds:
                continue
            end = min(end, max_seconds)
        segments.append(
            Segment(
                speaker=str(u[KEY_SPEAKER]),
                start=start,
                end=end,
                text=str(u[KEY_TEXT]),
            )
        )

    ref = Reference(uri=f"chime6_{session}", segments=segments)

    written_path: str | None = None
    if audio_out is not None and audio_in is not None:
        if max_seconds is None:
            raise ValueError(
                "audio_out requires max_seconds so the excerpt length is defined"
            )
        _slice_audio(audio_in, audio_out, duration=max_seconds)
        written_path = audio_out

    return ref, written_path


def _slice_audio(src: str, dst: str, duration: float) -> None:
    """Write the first `duration` seconds of src wav to dst.

    Uses soundfile for lossless float32 round-trip at the original sample rate.
    Reads only the required number of frames — does not load the full session
    into memory (CHiME-6 sessions are 2+ hours at 16 kHz).
    """
    import soundfile as sf

    with sf.SoundFile(src) as f:
        # Read at most n_frames from the start; honour the file's actual rate.
        actual_rate = f.samplerate
        n_frames_actual = int(duration * actual_rate)
        data = f.read(frames=n_frames_actual, dtype="float32", always_2d=False)

    sf.write(dst, data, actual_rate, subtype="FLOAT")
