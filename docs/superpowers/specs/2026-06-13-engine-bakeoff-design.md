# Engine Bake-off — Design Spec

**Date:** 2026-06-13
**Status:** Approved for planning
**Author:** revival session (accuracy-first track)

## 1. Purpose

Decide, on evidence, which ASR + diarization engine stack gives Tathurell the best
**speaker-attributed transcription accuracy** for its real domain (multi-speaker
recordings: tabletop RPG sessions, podcasts), before committing to any pipeline rewrite.

This project's deliverable is a **numbers table + a recommendation memo**. It is *not* a
pipeline rewrite. Once a winning stack is chosen, the rewrite gets its own spec.

### Why a benchmark corpus instead of hand-curated references
Hand-transcribing our own clips for ground truth is fragile: small, subjective, slow to
extend, and not reproducible. Instead we score against open corpora that already ship
**ground-truth transcripts with speaker labels**, which lets us compute speaker-attributed
metrics directly and re-run the eval at will.

## 2. Engines under test (3 stacks)

| Stack | ASR | Word timing | Diarization | Word→speaker | Role |
|---|---|---|---|---|---|
| **baseline** | vosk (gigaspeech) | vosk word times | pyannote 3.1 | max-overlap | current pipeline, the bar to beat |
| **whisperx** | faster-whisper (large-v3) | wav2vec2 forced alignment | pyannote (built in) | whisperx built-in | accuracy contender |
| **mlx-whisper** | mlx-whisper (large-v3) | whisper word times | pyannote 3.1 | max-overlap | speed contender on Apple Silicon |

Notes:
- The **baseline** uses the *fixed* current logic (post 2026-06-13 fixes), upgraded from the
  "first turn" heuristic to **max-overlap** attribution so the comparison isolates the engine,
  not a known-bad attribution rule.
- Diarization is pyannote 3.1 in two of three stacks (whisperx also uses pyannote under the
  hood), so DER is expected to be near-constant; the real differentiators are WER and cpWER.
- We deliberately do **not** test faster-whisper standalone — whisperx subsumes it and adds
  alignment + diarization.

## 3. Metrics

All computed against corpus ground truth:

- **cpWER** (concatenated minimum-permutation WER) via `meeteval` — **headline metric**. It
  scores exactly what Tathurell outputs: per-speaker text. Captures ASR *and* speaker
  attribution together.
- **WER** via `jiwer` — ASR quality alone (speaker-agnostic).
- **DER** (diarization error rate) via `pyannote.metrics` — diarization alone. Diagnostic.
- **Runtime** (wall-clock) and **peak memory** per run — practicality on the target Mac.

### Fairness: shared text normalization (non-negotiable)
vosk emits lowercase, no punctuation; Whisper emits cased, punctuated text with spelled/
digit number variation. Comparing WER without normalization is meaningless. **Every** engine
output and **every** reference passes through one shared normalizer — Whisper's
`EnglishTextNormalizer` (from `whisper.normalizers`) — before any metric is computed. The same
normalizer is applied to references so no stack is advantaged.

## 4. Data

Open corpora, all of which include ground-truth transcripts **with speaker labels**:

| Corpus | Domain / quality | License | Access |
|---|---|---|---|
| **AMI** | Meeting room; headset (clean) + far-field array (noisy) | CC BY 4.0 | OpenSLR / HuggingFace |
| **Earnings-21** | Earnings conference calls; telephony, entity-dense proper names | per repo `LICENSE.md` (CONFIRM before use) | GitHub `revdotcom/speech-datasets`, HF `Revai/earnings21` |
| **CHiME-6** | Real dinner parties; worn binaural + far-field Kinect arrays, noisy/overlapping | CC BY-SA 4.0 | OpenSLR #150 |

These cover the requested criteria: multiple speakers (all), transcript-with-diarization
(all), heterogeneous audio quality (AMI headset-vs-farfield, Earnings telephony, CHiME-6 noisy
home).

### Slicing — small dev subsets only
Never download/process full corpora. Target tens of minutes per corpus:
- AMI: ~1–2 meetings from the official **dev** split.
- Earnings-21: ~2–3 calls.
- CHiME-6: ~1–2 sessions from the **dev** split (avoid the 97 GB train tarball; pull only what
  the chosen sessions need).

The exact file IDs are pinned in the corpus loaders so the slice is reproducible. If a slice
is bounded for cost, the loader **logs what it dropped** (no silent truncation).

### Audio condition
Primary condition is the **realistic far-field / single distant channel** (AMI SDM, CHiME-6
distant array, Earnings call audio as-is) — closest to "someone recorded a session." The clean
headset/close-talk condition (AMI, CHiME-6 worn mics) is run **optionally** as an accuracy
ceiling, clearly labeled.

### Common reference format
Each corpus loader converts native annotations into one shared structure:
- ordered segments `{speaker, start, end, text}`,
- an RTTM (for DER),
- per-speaker concatenated text (for cpWER).
Engine adapters never see native corpus formats; metrics never see engine-native formats.

## 5. Architecture

A standalone evaluation package, isolated from the production `tathurell_transcribe.py`.
Each unit has one purpose and a uniform interface so it can be understood and tested alone.

```
eval/
  corpora/          # one loader per corpus -> common reference format
    ami.py
    earnings21.py
    chime6.py
    base.py         # shared reference dataclasses + RTTM/per-speaker-text helpers
  engines/          # one adapter per stack, all the same interface
    vosk_pyannote.py
    whisperx_stack.py
    mlxwhisper_pyannote.py
    base.py         # Engine protocol + shared max-overlap attribution helper
  metrics/
    normalize.py    # the single shared EnglishTextNormalizer wrapper
    wer.py          # jiwer
    cpwer.py        # meeteval
    der.py          # pyannote.metrics
  run_bakeoff.py    # orchestrates engines x slices -> results table + saved transcripts
  README.md         # how to run, how to add an engine/corpus
```

### Interfaces
- **Engine adapter:** `transcribe(audio_path) -> list[Word]` where
  `Word = {word: str, start: float, end: float, speaker: str}`. (Whisperx returns this
  natively; vosk/mlx adapters run pyannote + the shared max-overlap helper to fill `speaker`.)
- **Corpus loader:** `load(slice_id) -> Reference` (segments + RTTM path + per-speaker text).
- **Metric:** `score(hypothesis, reference) -> float`, hypothesis/reference already normalized.

### Shared attribution helper (max-overlap)
For stacks that diarize separately from ASR (baseline, mlx-whisper): assign each word to the
diarization turn with the **greatest temporal overlap** with the word interval
`[word.start, word.end]`; if a word overlaps no turn (silence gap), assign the nearest turn by
midpoint distance. This replaces the original "first turn that hasn't ended" heuristic and is
documented as the intended attribution rule.

## 6. Environment & risks

- New dependencies: `whisperx`, `faster-whisper`, `mlx-whisper`, `openai-whisper` (for the
  normalizer), `jiwer`, `meeteval`, `pyannote.metrics`, `huggingface_hub`/`datasets`.
- **Isolation (key mitigation):** the bake-off runs in a **separate virtualenv**
  (`tathurell-eval`), NOT the working `tathurell-reqs` env. Rationale: whisperx pins specific
  `torch`/`ctranslate2` versions that may conflict with the existing pyannote/torch install;
  the live pipeline must not be disturbed by eval deps.
- **MPS limitation:** faster-whisper (ctranslate2) has no MPS backend, so WhisperX runs on CPU
  on this Mac. mlx-whisper is Apple-Silicon-native (fast). pyannote runs on MPS. The CPU/MPS
  tradeoff is part of what the runtime metric measures, not a blocker.
- **HF gated models:** pyannote requires an `HF_TOKEN` with the model license accepted (same as
  the production script). The leaked token must be revoked; eval uses a fresh token via env.
- **Earnings-21 license:** confirm `LICENSE.md` terms permit this use before committing the
  slice; if not, drop Earnings-21 and proceed with AMI + CHiME-6.

## 7. Decision rule

**Winner = lowest mean cpWER across the three corpora in the realistic (far-field/telephony)
condition.** Runtime and peak memory are the practicality tiebreaker — a stack that is only
marginally more accurate but far slower/heavier on the target Mac may lose. The output is a
short recommendation memo containing the full results table (engine × corpus → WER, cpWER, DER,
runtime, memory) plus the saved transcripts for qualitative inspection.

## 8. Out of scope (deferred to later specs)

Pipeline rewrite, packaging/`requirements.txt`, configuration system, speaker-naming UX,
built-in transcoding, the dropped-`FinalResult()` tail in the production script, evaluating
pyAudioAnalysis (pyannote 3.1 is current SOTA; that old TODO is superseded). These wait until
the engine decision is made.
