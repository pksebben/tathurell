# WhisperX Pipeline Rewrite — Design Spec

**Date:** 2026-06-14
**Status:** Approved for planning
**Basis:** Bake-off result `2026-06-13-bakeoff-results.md` — WhisperX won every accuracy metric on every corpus.

## 1. Purpose

Replace Tathurell's transcription engine. `tathurell_transcribe.py` currently does
pyannote diarization + vosk (gigaspeech) ASR + per-word speaker attribution. The bake-off
showed WhisperX (Whisper large-v3 + forced alignment + integrated pyannote diarization +
word-level speaker assignment) is substantially more accurate. This rewrite swaps in WhisperX
and removes the hardcoded paths the swap makes obsolete, while keeping the existing interactive
speaker-naming flow and the `<audio>.transcription.txt` output unchanged.

**Scope:** engine swap + clean config. **Not** in scope (stays on the roadmap): speaker-naming
UX improvements, built-in transcoding, packaging.

## 2. Environment

The tool runs in the existing **`tathurell-eval`** pyenv venv, which already runs WhisperX +
pyannote.audio 4.0.4 end-to-end (proven by the bake-off). `.python-version` is updated from
`tathurell-reqs` to `tathurell-eval`. vosk is no longer a dependency of the tool (it remains in
the env harmlessly). The old `tathurell-reqs` env is retired (not deleted by this work).

## 3. Architecture

Three small units, each independently understandable and testable:

### `tathurell/whisperx_core.py` — the transcription engine
A class `WhisperXTranscriber` that loads models once and transcribes:
- `__init__(self, model="large-v3", device="cpu", token=None)`: loads the WhisperX ASR model and
  the pyannote diarization pipeline once. `token` defaults to the resolved `HF_TOKEN` (see §5).
  Device is `"cpu"` — faster-whisper (ctranslate2) has no MPS backend on Apple Silicon.
- `transcribe(self, audio_path) -> list[Word]` where `Word = {"word": str, "start": float,
  "end": float, "speaker": str | None}`: load audio → `model.transcribe` → `load_align_model` +
  `align` (the alignment model is loaded per call, keyed by the detected language) → diarize →
  `assign_word_speakers` → flatten segments to words, skipping tokens that alignment left without
  timestamps.

This is the bake-off's `eval/engines/whisperx_stack.py` logic promoted to an owned module. The
eval adapter is refactored to import/subclass `WhisperXTranscriber` so there is ONE WhisperX
implementation (DRY) and the bake-off stays reproducible.

Version facts already established (must be preserved): `whisperx.diarize.DiarizationPipeline`
(not top-level); pyannote 4.x auth arg is `token=`; pyannote 4.x `assign_word_speakers`/pipeline
returns are handled as in the working adapter.

### `tathurell/naming.py` — pure grouping + naming (no models)
- `group_by_speaker(words) -> list[{"speaker": str|None, "text": str}]`: collapse consecutive
  same-speaker words into one run; the word that triggers a speaker change starts the new run
  (the bug the old code had). Empty input → empty list.
- `apply_names(groups, names) -> str`: render `"{name}: {text}"` lines joined by newlines,
  where `names` maps speaker label → chosen name.

### `tathurell_transcribe.py` — thin CLI
1. Parse args: `audio_path` (positional), `--output` (default `<audio_path>.transcription.txt`),
   `--model` (default `large-v3`).
2. Resolve `HF_TOKEN` (§5); exit with a clear message if absent.
3. `words = WhisperXTranscriber(model=...).transcribe(audio_path)`.
4. `groups = group_by_speaker(words)`.
5. Interactive naming: for each distinct speaker (first appearance order), print the speaker's
   first chunk of text and `input("name: ")`; reuse the answer for that speaker. Resilient to
   EOF/closed stdin — fall back to the raw speaker label as the name (so non-interactive runs
   don't crash).
6. Write `apply_names(groups, names)` to the output path.

## 4. Data flow

`audio → WhisperXTranscriber.transcribe → words → group_by_speaker → interactive naming → apply_names → write <audio>.transcription.txt`

## 5. Configuration

All hardcoded paths are removed. Resolution order for the HF token:
1. `HF_TOKEN` environment variable.
2. Fallback: `eval/data/.hf_token` if it exists (convenience — same file the bake-off uses).
3. Else: exit non-zero with a message explaining how to set it and which pyannote model gates
   to accept (`speaker-diarization-3.1`, `segmentation-3.0`, `speaker-diarization-community-1`).

Audio input and output path come from CLI args. `--model` allows overriding `large-v3` (e.g. a
smaller/faster model) without code changes.

## 6. Error handling

- Missing token → clear exit (§5.3).
- WhisperX yields zero words (silence/failure) → write an empty output file and warn on stderr;
  do not crash.
- Interactive naming under closed/empty stdin (EOFError) → use the speaker label as the name.
- Audio path missing/unreadable → let WhisperX's load raise; surface the message.

## 7. Testing

- **Pure unit tests** for `naming.py` (`group_by_speaker`, `apply_names`) with synthetic word
  lists — no models, fast. Cover: speaker change keeps the triggering word; single speaker;
  empty input; name application formatting.
- **Token-gated smoke test** for `whisperx_core.transcribe` on `dollop_test_a.mp3` (skips
  without `HF_TOKEN`, like the eval smokes) — asserts a non-trivial word count with speakers.
- Run in the `tathurell-eval` venv. The eval suite must still pass after the adapter refactor.

## 8. What is removed

vosk ASR, the 4000-sample chunked decode loop, the manual per-word speaker attribution
(first-turn / max-overlap), the `FinalResult()` tail handling, and all hardcoded model/token
paths. These are superseded by WhisperX's integrated pipeline.

## 9. Migration / compatibility

Output format (`"{name}: {text}"` lines, `<audio>.transcription.txt`) is unchanged, so existing
transcripts and downstream expectations are unaffected. The CLI gains `--output`/`--model` flags
but the positional `audio_path` call form still works.
