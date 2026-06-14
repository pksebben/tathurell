"""AMI loader -> Reference, using the HuggingFace `edinburghcstr/ami` dataset.

Pins a single validation meeting as the reproducible slice. Far-field (sdm) is
the realistic primary condition; headset (ihm) is an optional clean ceiling.

Audio note: HF rows are per-utterance AudioDecoder objects (torchcodec), not a
single meeting wav. The audio_out branch reconstructs a meeting-aligned buffer by
placing each utterance at its begin_time offset so that the audio timeline matches
the segment begin_time/end_time values in the Reference — essential for any
downstream engine test. The buffer is zero-filled (silence) in gaps.

Schema (edinburghcstr/ami, confirmed by introspection 2026-06-13):
  meeting_id, audio_id, text, audio, begin_time, end_time, microphone_id, speaker_id
"""
from __future__ import annotations

import numpy as np
from datasets import load_dataset

from eval.corpora.base import Reference, Segment

KEY_MEETING = "meeting_id"
KEY_SPEAKER = "speaker_id"
KEY_TEXT = "text"
KEY_START = "begin_time"
KEY_END = "end_time"

# Fixed validation meeting used as the reproducible slice.
# ES2011a: 364 utterances, 4 speakers, ~18-min span. Config: sdm (far-field).
AMI_MEETING_ID = "ES2011a"

_SAMPLE_RATE = 16000  # AMI HF dataset always resamples to 16 kHz


def load(
    condition: str = "sdm",
    meeting_id: str = AMI_MEETING_ID,
    audio_out: str | None = None,
) -> tuple[Reference, str | None]:
    """Load a single AMI meeting from the HF dataset validation split.

    Uses streaming so only the target meeting rows are fetched. Iterates the
    stream and stops as soon as the meeting ends (meetings are contiguous),
    avoiding a full-split download.

    Args:
        condition:  Dataset config name; "sdm" (far-field) or "ihm" (headset).
        meeting_id: Which meeting to extract; default is the pinned slice.
        audio_out:  If given, write a meeting-aligned mono WAV to this path.
                    The audio timeline matches segment begin_time/end_time values.

    Returns:
        (Reference, audio_out_path_or_None)
    """
    ds = load_dataset(
        "edinburghcstr/ami",
        condition,
        split="validation",
        streaming=True,
    )

    rows: list[dict] = []
    past_meeting = False
    for row in ds:
        if row[KEY_MEETING] == meeting_id:
            rows.append(row)
            past_meeting = True
        elif past_meeting:
            # Meetings are contiguous in the stream; once we leave, stop.
            break

    if not rows:
        raise ValueError(
            f"meeting {meeting_id!r} not found in AMI/{condition} validation split"
        )

    segments = [
        Segment(
            speaker=str(r[KEY_SPEAKER]),
            start=float(r[KEY_START]),
            end=float(r[KEY_END]),
            text=str(r[KEY_TEXT]),
        )
        for r in rows
    ]

    if audio_out is not None:
        _write_meeting_audio(rows, audio_out)

    return Reference(uri=f"ami_{condition}_{meeting_id}", segments=segments), audio_out


def _write_meeting_audio(rows: list[dict], path: str) -> None:
    """Reconstruct a meeting-aligned mono WAV from per-utterance AudioDecoder rows.

    Each utterance's decoded audio is placed at its begin_time offset within a
    zero-filled buffer whose length = ceil(max(end_time)) * sample_rate. Overlapping
    utterances are summed then clipped to [-1, 1]. The result is written as float32.
    """
    import sys

    import soundfile as sf

    # Determine meeting duration from reference times (seconds).
    max_end = max(float(r[KEY_END]) for r in rows)
    total_samples = int(np.ceil(max_end * _SAMPLE_RATE))
    buffer = np.zeros(total_samples, dtype=np.float32)

    for r in rows:
        audio_decoder = r["audio"]
        try:
            audio_samples = audio_decoder.get_all_samples()
            # data shape: [channels, n_samples]; take channel 0 as mono
            arr = audio_samples.data.numpy()[0]
        except Exception as exc:
            # If a single utterance fails to decode, skip it — don't abort the
            # whole meeting. Log to stderr so the caller can see it.
            print(f"[ami] WARNING: failed to decode utterance {r.get('audio_id')}: {exc}", file=sys.stderr)
            continue

        start_sample = int(float(r[KEY_START]) * _SAMPLE_RATE)
        end_sample = start_sample + len(arr)

        if end_sample > total_samples:
            # Truncate if slightly over (floating-point rounding edge).
            arr = arr[: total_samples - start_sample]
            end_sample = total_samples

        buffer[start_sample:end_sample] += arr

    # Clip overlapping regions to valid float32 range.
    np.clip(buffer, -1.0, 1.0, out=buffer)

    sf.write(path, buffer, _SAMPLE_RATE, subtype="FLOAT")
