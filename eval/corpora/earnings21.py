"""Earnings-21 loader -> Reference. Telephony/conference audio with dense proper
names and multi-speaker earnings calls.

Sources:
  Audio:  HuggingFace `Revai/earnings21` dataset, split="test".
          Each HF row is ONE full call (per-call AudioDecoder) — NOT per-utterance.
          Audio is written directly (no reconstruction needed).

  Text:   GitHub revdotcom/speech-datasets earnings21/transcripts/nlp_references/
          <file_id>.nlp — tab(|)-delimited token table with columns:
            token | speaker | ts | endTs | punctuation | case | tags | wer_tags
          The ts/endTs columns are ALWAYS empty in this dataset; timing is NOT
          available at the token level.

  RTTM:   GitHub earnings21/rttms/<file_id>.rttm provides per-segment speaker
          timing, but uses a DIFFERENT speaker-ID scheme than the NLP files.
          The two cannot be reliably joined (speaker-ID sets overlap only partially
          and per-speaker turn counts differ). RTTM data is NOT used here.

Timing caveat: Because no per-segment timestamps are available, all Segment
start/end values are set to (0.0, audio_length). DER is therefore NOT computable
from this Reference. WER and cpWER are fully supported.

Schema (HF dataset, confirmed by introspection 2026-06-13):
  audio, audio_length, original_sample_rate, company_name, financial_quarter,
  sector, speaker_switches, unique_speakers, curator_id, text

Pinned call: file_id 4386541 (shortest call in the dataset with multiple speakers).
  ~1097 s (~18.3 min), 5 speakers, 17 speaker turns.
  Previously pinned to 4341191 (GE Q1 2020, ~5741s) — too long for CPU WhisperX.
"""
from __future__ import annotations

import urllib.error
import urllib.request

from datasets import load_dataset

from eval.corpora.base import Reference, Segment

# GitHub base URL for the speech-datasets repo (earnings21 NLP files).
_GITHUB_RAW = (
    "https://raw.githubusercontent.com/revdotcom/speech-datasets/main/earnings21"
)

# Pinned call: shortest multi-speaker call in the dataset.
# file_id 4386541: ~1097s (~18.3 min), 5 speakers, 17 speaker turns.
# Identified by sorting earnings21-file-metadata.csv by audio_length ascending.
# Re-pinned from 4341191 (GE Q1 2020, ~5741s) for CPU-tractable WhisperX runs.
EARNINGS_CALL_ID = "4386541"

# Tolerance (seconds) for the audio_length cross-check between the metadata CSV
# and the HF dataset row. If the two differ by more than this, the CSV/HF ordering
# assumption is violated and we raise rather than return wrong-call data.
_AUDIO_LENGTH_TOL = 2.0


def load(
    call_id: str = EARNINGS_CALL_ID,
    audio_out: str | None = None,
) -> tuple[Reference, str | None]:
    """Load a single Earnings-21 call as a Reference.

    Speaker turns are reconstructed from the GitHub NLP token file (speaker +
    text). Timing is NOT available; all Segment start/end values are set to
    (0.0, audio_length). DER is not computable from this Reference; WER and
    cpWER are fully supported.

    Args:
        call_id:   Earnings-21 file_id (numeric string). Default is the pinned
                   slice (4386541, ~18.3 min, 5 speakers).
        audio_out: If given, write the call audio as a WAV to this path.

    Returns:
        (Reference, audio_out_path_or_None)
    """
    # --- 1. Load NLP token file from GitHub to get speaker turns + text. ---
    turns = _load_nlp_turns(call_id)
    if not turns:
        raise ValueError(f"No tokens found in NLP file for call {call_id}")

    # --- 2. Fetch audio_length from the HF dataset row (needed for Segment end). ---
    audio_length, audio_row = _find_hf_row(call_id)

    # --- 3. Build Segments. ---
    # Timing unavailable: start=0.0, end=audio_length for all segments.
    # This reflects the limitation clearly rather than inventing timing.
    segments = [
        Segment(
            speaker=str(spk),
            start=0.0,
            end=audio_length,
            text=text,
        )
        for spk, text in turns
    ]

    # --- 4. Optionally write call audio. ---
    if audio_out is not None:
        _write_call_audio(audio_row, audio_out)

    return Reference(uri=f"earnings21_{call_id}", segments=segments), audio_out


def _load_nlp_turns(call_id: str) -> list[tuple[str, str]]:
    """Fetch and parse the GitHub NLP token file into speaker turns.

    Returns a list of (speaker_id, text) tuples in token order. Consecutive
    tokens from the same speaker are merged into one turn. Punctuation tokens
    (column 4) are appended directly to the preceding token without a space.
    """
    url = f"{_GITHUB_RAW}/transcripts/nlp_references/{call_id}.nlp"
    try:
        raw = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise ValueError(
            f"NLP file not found for call {call_id} (HTTP {exc.code}): {url}"
        ) from exc

    turns: list[tuple[str, str]] = []
    current_speaker: str | None = None
    current_tokens: list[str] = []

    for line in raw.splitlines():
        parts = line.split("|")
        if len(parts) < 2 or parts[0] == "token":
            # Skip header and malformed lines.
            continue
        token = parts[0]
        speaker = parts[1]
        punct = parts[4] if len(parts) > 4 else ""

        if speaker != current_speaker:
            if current_tokens and current_speaker is not None:
                turns.append((current_speaker, " ".join(current_tokens)))
            current_speaker = speaker
            current_tokens = []

        # Punctuation is appended to the preceding token without a space.
        if punct and current_tokens:
            current_tokens[-1] = current_tokens[-1] + punct
        current_tokens.append(token)

    if current_tokens and current_speaker is not None:
        turns.append((current_speaker, " ".join(current_tokens)))

    return turns


def _find_hf_row(call_id: str) -> tuple[float, dict]:
    """Stream the HF dataset to locate the row matching call_id.

    Matching is done by position: the HF test split and the metadata CSV share
    the same sort order. The metadata CSV is fetched from GitHub to find the
    0-based index of call_id AND its expected audio_length. The HF stream is
    then advanced to that position, and the audio_length from the HF row is
    compared against the CSV value (within _AUDIO_LENGTH_TOL). A mismatch
    means the CSV/HF ordering assumption is violated — we raise instead of
    silently returning wrong-call data.

    Returns (audio_length, row_dict).

    Raises ValueError if call_id is not found or the audio_length cross-check fails.
    """
    meta_url = (
        "https://raw.githubusercontent.com/revdotcom/speech-datasets/main/"
        "earnings21/earnings21-file-metadata.csv"
    )
    meta_raw = urllib.request.urlopen(meta_url, timeout=30).read().decode("utf-8")
    meta_lines = meta_raw.strip().splitlines()

    # Parse header to locate the audio_length column by name (don't assume index).
    header = [col.strip() for col in meta_lines[0].split(",")]
    try:
        audio_length_col = header.index("audio_length")
    except ValueError as exc:
        raise ValueError(
            f"'audio_length' column not found in earnings21 metadata CSV header: {header}"
        ) from exc

    call_index: int | None = None
    expected_length: float | None = None
    for i, line in enumerate(meta_lines[1:], start=0):
        cols = line.split(",")
        if cols[0].strip() == call_id:
            call_index = i
            expected_length = float(cols[audio_length_col].strip())
            break

    if call_index is None or expected_length is None:
        raise ValueError(
            f"call {call_id} not found in earnings21 metadata CSV"
        )

    # Stream HF dataset to the target index.
    ds = load_dataset("Revai/earnings21", split="test", streaming=True)
    for idx, row in enumerate(ds):
        if idx == call_index:
            actual_length = float(row["audio_length"])
            # Verify that the HF row at this position matches the call we expected.
            # A large discrepancy means the CSV and HF stream are out of sync.
            if abs(actual_length - expected_length) > _AUDIO_LENGTH_TOL:
                raise ValueError(
                    f"HF/CSV ordering assumption violated for call {call_id}: "
                    f"CSV audio_length={expected_length:.3f}s but HF row at index "
                    f"{call_index} has audio_length={actual_length:.3f}s "
                    f"(tolerance={_AUDIO_LENGTH_TOL}s)"
                )
            return actual_length, row

    raise ValueError(
        f"HF dataset ended before reaching index {call_index} for call {call_id}"
    )


def _write_call_audio(row: dict, path: str) -> None:
    """Decode and write call-level audio to a WAV file.

    Audio in the HF dataset is a single AudioDecoder per call — not per utterance.
    No temporal reconstruction is needed; the decoded array is written directly.
    """
    import numpy as np
    import soundfile as sf

    audio_decoder = row["audio"]
    samples = audio_decoder.get_all_samples()
    # data shape: [channels, n_samples]; collapse to mono by taking channel 0.
    arr = samples.data.numpy()[0].astype(np.float32)
    sr = int(samples.sample_rate)
    sf.write(path, arr, sr, subtype="FLOAT")
