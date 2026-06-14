# eval — Engine Bake-Off

Benchmarks three ASR+diarization engines across three corpora, reporting WER,
cpWER (headline metric), DER, runtime, and peak memory.

Full design: [`docs/superpowers/specs/2026-06-13-engine-bakeoff-design.md`](../docs/superpowers/specs/2026-06-13-engine-bakeoff-design.md)

---

## Quick Start

```bash
# Option A — token in environment
export HF_TOKEN=hf_...
python -m eval.run_bakeoff

# Option B — token in file (gitignored)
echo "hf_..." > eval/data/.hf_token
python -m eval.run_bakeoff
```

The script prints a markdown results table to stdout and saves each engine's
transcript to `eval/data/<uri>.<engine>.txt` for inspection.

---

## Virtual Environment

All eval dependencies live in a **separate venv** to avoid polluting the
main Tathurell environment:

```
~/.pyenv/versions/tathurell-eval/
```

Always activate or call Python explicitly:

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m eval.run_bakeoff
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/
```

### meeteval install workaround (macOS / clang)

meeteval 0.4.3 includes a Cython extension that fails to compile on macOS
without a minimum deployment target set. Set this env var **during pip
install**:

```bash
MACOSX_DEPLOYMENT_TARGET=11.0 pip install meeteval==0.4.3
```

---

## Headline Metric

**cpWER (concatenated minimum-permutation WER)** — measures ASR accuracy and
speaker attribution together under the best speaker assignment. Lower is better.

Runtime is the tiebreaker: a model that achieves the same cpWER in half the
time wins for real-time applications.

---

## Corpora and Budget

| Corpus | Condition | Budget | DER? | Notes |
|---|---|---|---|---|
| AMI | SDM (far-field) | first 600 s | Yes | Meeting ES2011a, 4 speakers |
| Earnings-21 | telephony | full call | **N/A** | Call 4386541, ~18 min, 5 speakers — no per-segment timestamps |
| CHiME-6 | far-field dinner party | first 600 s | Yes | Session S02 |

DER is **N/A for Earnings-21** because the NLP token files carry no per-segment
timestamps — all segments are stored as `(0.0, audio_length)`. WER and cpWER
are unaffected by this limitation.

AMI and CHiME-6 are excerpted to 600 s (`BUDGET_SECONDS`) so CPU-bound engines
(WhisperX runs on CPU on Apple Silicon) complete in a reasonable wall-clock time.

---

## Engines

| Engine | File |
|---|---|
| VoskPyannote | `eval/engines/vosk_pyannote.py` |
| WhisperXStack | `eval/engines/whisperx_stack.py` |
| MlxWhisperPyannote | `eval/engines/mlxwhisper_pyannote.py` |

All engines implement `name: str` and `transcribe(audio_path) -> list[dict]`,
where each dict has `{"word", "start", "end", "speaker"}`.

---

## Adding an Engine

1. Create `eval/engines/<your_engine>.py` implementing `name` and `transcribe`.
2. Import and add it to the `engines` list in `eval/run_bakeoff.py`.
3. Add a smoke test in `tests/test_<your_engine>_smoke.py` (token-gated).

## Adding a Corpus

1. Create `eval/corpora/<corpus>.py` returning `(Reference, audio_path)`.
2. Add a slice dict to the `slices` list in `run()` — set `compute_der=True`
   if the Reference has real per-segment timestamps, `False` otherwise.
3. Add a smoke test in `tests/test_<corpus>_smoke.py`.
