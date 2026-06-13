# Engine Bake-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated evaluation harness that scores three ASR+diarization engine stacks (vosk+pyannote, WhisperX, mlx-whisper+pyannote) against open ground-truth corpora (AMI, Earnings-21, CHiME-6) and produces a WER/cpWER/DER + runtime table to decide Tathurell's pipeline engine.

**Architecture:** A standalone `eval/` package, run from a separate `tathurell-eval` virtualenv so it never disturbs the working pipeline. Pure-logic units (reference format, max-overlap attribution, metric wrappers) are built test-first. Engine adapters and corpus loaders depend on heavy models/data and external formats, so they are built introspect-first and verified by smoke runs on tiny slices. All metrics run through one shared text normalizer for fairness.

**Tech Stack:** Python 3.10, pytest, vosk, pyannote.audio 3.1, whisperx (faster-whisper/ctranslate2), mlx-whisper, openai-whisper (normalizer), jiwer, meeteval, pyannote.metrics, huggingface_hub/datasets.

Spec: `docs/superpowers/specs/2026-06-13-engine-bakeoff-design.md`.

---

## File structure

```
eval/
  __init__.py
  corpora/
    __init__.py
    base.py            # Segment/Reference dataclasses + RTTM / per-speaker-text helpers (PURE)
    ami.py             # AMI dev-slice loader -> Reference
    earnings21.py      # Earnings-21 loader -> Reference  (license-gated)
    chime6.py          # CHiME-6 dev-slice loader -> Reference
  engines/
    __init__.py
    base.py            # Word/Turn types + Engine protocol + max-overlap attribution (PURE)
    vosk_pyannote.py   # baseline adapter (current pipeline logic, fixed)
    whisperx_stack.py  # whisperx adapter
    mlxwhisper_pyannote.py  # mlx-whisper + pyannote adapter
  metrics/
    __init__.py
    normalize.py       # shared EnglishTextNormalizer wrapper (PURE)
    wer.py             # jiwer wrapper (PURE)
    cpwer.py           # meeteval wrapper (PURE)
    der.py             # pyannote.metrics wrapper (PURE)
  run_bakeoff.py       # orchestrator: engines x slices -> results table + saved transcripts
  README.md
  conftest.py          # pytest fixtures (synthetic words/turns/segments)
tests/                 # mirrors eval/ ; pure-logic tests live here
```

---

## Task 1: Scaffold venv and package skeleton

**Files:**
- Create: `eval/__init__.py`, `eval/corpora/__init__.py`, `eval/engines/__init__.py`, `eval/metrics/__init__.py`
- Create: `eval/requirements-eval.txt`
- Create: `tests/__init__.py`, `eval/conftest.py`
- Create: `pytest.ini`

- [ ] **Step 1: Create the eval virtualenv (separate from tathurell-reqs)**

```bash
pyenv virtualenv 3.10.7 tathurell-eval
# pin it only for the eval dir usage; do NOT overwrite the repo .python-version
~/.pyenv/versions/tathurell-eval/bin/python -m pip install --upgrade pip
```

Expected: env `tathurell-eval` created. Use its python explicitly as `EVAL_PY=~/.pyenv/versions/tathurell-eval/bin/python` for all eval steps.

- [ ] **Step 2: Write `eval/requirements-eval.txt`**

```
vosk
pydub
numpy
torch
torchaudio
pyannote.audio>=3.1
pyannote.metrics
whisperx
mlx-whisper
openai-whisper
jiwer
meeteval
huggingface_hub
datasets
soundfile
pytest
```

- [ ] **Step 3: Install (long; ctranslate2/torch are large)**

```bash
~/.pyenv/versions/tathurell-eval/bin/pip install -r eval/requirements-eval.txt
```

Expected: completes. If whisperx forces an incompatible torch and breaks the install, note the resolved versions and continue — version reconciliation is part of Task 12's introspection.

- [ ] **Step 4: Create empty package files**

Create `eval/__init__.py`, `eval/corpora/__init__.py`, `eval/engines/__init__.py`, `eval/metrics/__init__.py`, `tests/__init__.py` — each an empty file.

- [ ] **Step 5: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -q
```

- [ ] **Step 6: Create `eval/conftest.py` with shared synthetic fixtures**

```python
import pytest


@pytest.fixture
def turns():
    # diarization turns: (speaker, start, end), chronological
    return [
        {"speaker": "A", "start": 0.0, "end": 5.0},
        {"speaker": "B", "start": 5.0, "end": 10.0},
        {"speaker": "A", "start": 10.0, "end": 15.0},
    ]


@pytest.fixture
def words():
    # asr words with absolute start/end
    return [
        {"word": "hello", "start": 0.5, "end": 1.0},
        {"word": "there", "start": 6.0, "end": 6.4},   # inside B
        {"word": "bye", "start": 16.0, "end": 16.5},   # past last turn -> nearest A
    ]
```

- [ ] **Step 7: Verify pytest runs (no tests yet)**

```bash
cd /Users/benmorsillo/code/Tathurell && ~/.pyenv/versions/tathurell-eval/bin/python -m pytest
```

Expected: "no tests ran" exit 5 (acceptable) or 0.

- [ ] **Step 8: Commit**

```bash
git add eval tests pytest.ini && git commit -m "eval: scaffold bake-off package and venv requirements"
```

---

## Task 2: Reference format + RTTM/per-speaker-text helpers (PURE, TDD)

**Files:**
- Create: `eval/corpora/base.py`
- Test: `tests/test_corpora_base.py`

- [ ] **Step 1: Write failing test**

```python
from eval.corpora.base import Segment, Reference


def test_per_speaker_text_groups_and_orders_by_time():
    ref = Reference(uri="m1", segments=[
        Segment(speaker="A", start=0.0, end=1.0, text="hello world"),
        Segment(speaker="B", start=1.0, end=2.0, text="hi"),
        Segment(speaker="A", start=2.0, end=3.0, text="again"),
    ])
    assert ref.per_speaker_text() == {"A": "hello world again", "B": "hi"}


def test_to_rttm_lines_format():
    ref = Reference(uri="m1", segments=[Segment("A", 0.0, 1.5, "x")])
    line = ref.to_rttm().strip()
    # SPEAKER <uri> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>
    assert line == "SPEAKER m1 1 0.000 1.500 <NA> <NA> A <NA> <NA>"
```

- [ ] **Step 2: Run test, verify it fails**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_corpora_base.py -v
```
Expected: FAIL (ImportError: cannot import name 'Segment').

- [ ] **Step 3: Implement `eval/corpora/base.py`**

```python
"""Common ground-truth reference format shared by all corpus loaders.

Corpus-native formats (AMI NXT XML, Earnings-21 token files, CHiME-6 JSON) are
converted into `Reference` here so engine adapters and metrics never touch a
corpus-specific schema. One reference per recording (uri).
"""
from dataclasses import dataclass, field


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class Reference:
    uri: str
    segments: list = field(default_factory=list)

    def per_speaker_text(self):
        """{speaker: concatenated text} in time order — input for cpWER reference."""
        ordered = sorted(self.segments, key=lambda s: s.start)
        out = {}
        for s in ordered:
            t = s.text.strip()
            if not t:
                continue
            out[s.speaker] = f"{out[s.speaker]} {t}" if s.speaker in out else t
        return out

    def to_rttm(self):
        """NIST RTTM text (one SPEAKER line per segment) — input for DER."""
        lines = []
        for s in sorted(self.segments, key=lambda s: s.start):
            dur = s.end - s.start
            lines.append(
                f"SPEAKER {self.uri} 1 {s.start:.3f} {dur:.3f} <NA> <NA> {s.speaker} <NA> <NA>"
            )
        return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test, verify pass**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_corpora_base.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/corpora/base.py tests/test_corpora_base.py && git commit -m "eval: reference format with RTTM and per-speaker-text"
```

---

## Task 3: Max-overlap speaker attribution (PURE, TDD)

**Files:**
- Create: `eval/engines/base.py`
- Test: `tests/test_engines_base.py`

- [ ] **Step 1: Write failing test**

```python
from eval.engines.base import assign_speakers_max_overlap


def test_word_assigned_to_max_overlap_turn(turns, words):
    out = assign_speakers_max_overlap(words, turns)
    assert [w["speaker"] for w in out] == ["A", "B", "A"]  # last word past turns -> nearest (A)


def test_overlap_picks_greater_overlap_not_first():
    turns = [
        {"speaker": "A", "start": 0.0, "end": 2.1},
        {"speaker": "B", "start": 2.0, "end": 5.0},
    ]
    # word [2.0,2.9]: 0.1s overlap with A, 0.9s with B -> B
    out = assign_speakers_max_overlap([{"word": "w", "start": 2.0, "end": 2.9}], turns)
    assert out[0]["speaker"] == "B"


def test_empty_turns_assigns_none():
    out = assign_speakers_max_overlap([{"word": "w", "start": 0.0, "end": 1.0}], [])
    assert out[0]["speaker"] is None
```

- [ ] **Step 2: Run test, verify it fails**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_engines_base.py -v
```
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `eval/engines/base.py`**

```python
"""Engine adapter contract + the shared word->speaker attribution rule.

A `Word` is {"word": str, "start": float, "end": float, "speaker": str|None}.
A `Turn` is {"speaker": str, "start": float, "end": float}.

Engines that diarize separately from ASR (vosk, mlx-whisper) call
`assign_speakers_max_overlap` to fill the speaker field. The rule: assign each
word to the diarization turn it overlaps most in time; if it overlaps no turn
(falls in a silence gap or past the last turn), assign the nearest turn by
midpoint distance. This replaces the original "first turn not yet ended"
heuristic, which ignored word end and turn start.
"""
from typing import Protocol


class Engine(Protocol):
    name: str

    def transcribe(self, audio_path: str) -> list:
        """Return a list of Word dicts with speaker filled in."""
        ...


def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_max_overlap(words, turns):
    for w in words:
        if not turns:
            w["speaker"] = None
            continue
        best = max(turns, key=lambda t: _overlap(w["start"], w["end"], t["start"], t["end"]))
        if _overlap(w["start"], w["end"], best["start"], best["end"]) > 0.0:
            w["speaker"] = best["speaker"]
        else:
            wm = (w["start"] + w["end"]) / 2.0
            nearest = min(turns, key=lambda t: abs(wm - (t["start"] + t["end"]) / 2.0))
            w["speaker"] = nearest["speaker"]
    return words
```

- [ ] **Step 4: Run test, verify pass**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_engines_base.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add eval/engines/base.py tests/test_engines_base.py && git commit -m "eval: max-overlap word-to-speaker attribution"
```

---

## Task 4: Shared text normalizer (PURE, TDD)

**Files:**
- Create: `eval/metrics/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write failing test**

```python
from eval.metrics.normalize import normalize_text


def test_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, World!") == "hello world"


def test_idempotent():
    once = normalize_text("Mr. Smith paid $5.")
    assert normalize_text(once) == once
```

- [ ] **Step 2: Run test, verify it fails**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_normalize.py -v
```
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `eval/metrics/normalize.py`**

```python
"""Single shared normalizer applied to EVERY engine output AND every reference
before any WER/cpWER computation, so engines that differ only in casing/
punctuation/number formatting (vosk: lowercase no-punct vs Whisper: cased,
punctuated) are compared fairly. Uses Whisper's EnglishTextNormalizer.
"""
from whisper.normalizers import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    return _normalizer(text)
```

- [ ] **Step 4: Run test, verify pass**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_normalize.py -v
```
Expected: PASS. (If `EnglishTextNormalizer` collapses "$5" to "5 dollars" or "5", the idempotence test still holds; the lowercase/punct test asserts the stable behavior.)

- [ ] **Step 5: Commit**

```bash
git add eval/metrics/normalize.py tests/test_normalize.py && git commit -m "eval: shared EnglishTextNormalizer wrapper"
```

---

## Task 5: WER metric (PURE, TDD)

**Files:**
- Create: `eval/metrics/wer.py`
- Test: `tests/test_wer.py`

- [ ] **Step 1: Write failing test**

```python
from eval.metrics.wer import word_error_rate


def test_identical_is_zero():
    assert word_error_rate("the quick brown fox", "the quick brown fox") == 0.0


def test_one_substitution_in_four_words():
    # normalization lowercases; one wrong word out of 4 -> 0.25
    assert word_error_rate("the quick brown fox", "the QUICK brown DOG") == 0.25
```

- [ ] **Step 2: Run test, verify it fails**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_wer.py -v
```
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `eval/metrics/wer.py`**

```python
"""Speaker-agnostic WER via jiwer, on normalized text. Reference first."""
import jiwer
from eval.metrics.normalize import normalize_text


def word_error_rate(reference: str, hypothesis: str) -> float:
    return jiwer.wer(normalize_text(reference), normalize_text(hypothesis))
```

- [ ] **Step 4: Run test, verify pass**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_wer.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/metrics/wer.py tests/test_wer.py && git commit -m "eval: WER metric on normalized text"
```

---

## Task 6: cpWER metric (PURE, TDD)

**Files:**
- Create: `eval/metrics/cpwer.py`
- Test: `tests/test_cpwer.py`

- [ ] **Step 1: Confirm meeteval entry point in the installed version**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -c "from meeteval.wer.wer.cp import cp_word_error_rate; print('ok')"
```
Expected: `ok`. If the import path differs, run `~/.pyenv/versions/tathurell-eval/bin/python -c "import meeteval; print(dir(meeteval.wer))"` and use the discovered `cp_word_error_rate` location in Step 3.

- [ ] **Step 2: Write failing test**

```python
from eval.metrics.cpwer import cp_wer


def test_perfect_match_zero():
    ref = {"A": "hello world", "B": "good morning"}
    hyp = {"A": "hello world", "B": "good morning"}
    assert cp_wer(reference=ref, hypothesis=hyp) == 0.0


def test_speaker_swap_still_matched_by_cpwer():
    # cpWER finds the best speaker permutation, so swapped labels with correct
    # words still score 0.
    ref = {"A": "hello world", "B": "good morning"}
    hyp = {"X": "good morning", "Y": "hello world"}
    assert cp_wer(reference=ref, hypothesis=hyp) == 0.0
```

- [ ] **Step 3: Implement `eval/metrics/cpwer.py`**

```python
"""Concatenated minimum-permutation WER (cpWER) via meeteval — the headline
metric. Scores per-speaker text against the reference under the best speaker
assignment, so it measures ASR + speaker attribution together. Both sides are
normalized first.
"""
from meeteval.wer.wer.cp import cp_word_error_rate
from eval.metrics.normalize import normalize_text


def cp_wer(reference: dict, hypothesis: dict) -> float:
    """reference/hypothesis: {speaker: concatenated text}."""
    ref = {spk: normalize_text(t) for spk, t in reference.items()}
    # meeteval accepts the hypothesis as a list of per-speaker strings.
    hyp = [normalize_text(t) for t in hypothesis.values()]
    return cp_word_error_rate(reference=ref, hypothesis=hyp).error_rate
```

- [ ] **Step 4: Run test, verify pass**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_cpwer.py -v
```
Expected: PASS. If `cp_word_error_rate` rejects a list hypothesis in this version, pass `hyp` as a dict `{i: text}` instead (per Step 1 discovery) and keep the test green.

- [ ] **Step 5: Commit**

```bash
git add eval/metrics/cpwer.py tests/test_cpwer.py && git commit -m "eval: cpWER metric via meeteval"
```

---

## Task 7: DER metric (PURE, TDD)

**Files:**
- Create: `eval/metrics/der.py`
- Test: `tests/test_der.py`

- [ ] **Step 1: Confirm pyannote.metrics + RTTM loader API**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -c "from pyannote.metrics.diarization import DiarizationErrorRate; from pyannote.database.util import load_rttm; print('ok')"
```
Expected: `ok`. If `load_rttm` is elsewhere, discover with `~/.pyenv/versions/tathurell-eval/bin/python -c "import pyannote.database.util as u; print([x for x in dir(u) if 'rttm' in x.lower()])"`.

- [ ] **Step 2: Write failing test**

```python
import textwrap
from eval.metrics.der import diarization_error_rate


def test_identical_rttm_zero(tmp_path):
    rttm = textwrap.dedent(
        """\
        SPEAKER m1 1 0.000 5.000 <NA> <NA> A <NA> <NA>
        SPEAKER m1 1 5.000 5.000 <NA> <NA> B <NA> <NA>
        """
    )
    ref = tmp_path / "ref.rttm"; ref.write_text(rttm)
    hyp = tmp_path / "hyp.rttm"; hyp.write_text(rttm)
    assert diarization_error_rate(str(ref), str(hyp)) == 0.0
```

- [ ] **Step 3: Implement `eval/metrics/der.py`**

```python
"""Diarization Error Rate via pyannote.metrics. Inputs are RTTM file paths.
Diagnostic only (pyannote is shared across most stacks), but useful to confirm
diarization isn't the differentiator.
"""
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm


def _single_annotation(rttm_path):
    annotations = load_rttm(rttm_path)  # {uri: Annotation}
    return next(iter(annotations.values()))


def diarization_error_rate(reference_rttm: str, hypothesis_rttm: str) -> float:
    metric = DiarizationErrorRate()
    return float(metric(_single_annotation(reference_rttm), _single_annotation(hypothesis_rttm)))
```

- [ ] **Step 4: Run test, verify pass**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_der.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/metrics/der.py tests/test_der.py && git commit -m "eval: DER metric via pyannote.metrics"
```

---

## Task 8: AMI corpus loader (primary; introspect + smoke)

**Files:**
- Create: `eval/corpora/ami.py`
- Test: `tests/test_ami_smoke.py`

- [ ] **Step 1: Introspect the AMI HuggingFace dataset schema**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -c "
from datasets import load_dataset
ds = load_dataset('edinburghcstr/ami', 'ihm', split='validation', streaming=True)
print(next(iter(ds)).keys())
"
```
Expected: prints feature keys (e.g. `meeting_id`, `audio`, `text`, `speaker_id`, `begin_time`, `end_time`, `microphone_id`). Record the exact key names; use them in Step 2. If `edinburghcstr/ami` config names differ, list configs with `load_dataset('edinburghcstr/ami')` error output or the dataset card. Pick one validation meeting id to use as the fixed slice; far-field config (`sdm`) is primary, `ihm` (headset) is the optional ceiling.

- [ ] **Step 2: Implement `eval/corpora/ami.py`**

```python
"""AMI loader -> Reference, using the HuggingFace `edinburghcstr/ami` dataset.
Pins a single validation meeting as the reproducible slice. Far-field (sdm) is
the realistic primary condition; headset (ihm) is an optional clean ceiling.

NOTE: field names below are taken from the schema printed in Task 8 Step 1.
Adjust the four KEY_* constants if introspection shows different names.
"""
from datasets import load_dataset
from eval.corpora.base import Segment, Reference

KEY_MEETING = "meeting_id"
KEY_SPEAKER = "speaker_id"
KEY_TEXT = "text"
KEY_START = "begin_time"
KEY_END = "end_time"

# pinned reproducible slice (set from Step 1 introspection)
AMI_MEETING_ID = "EN2002a"


def load(condition: str = "sdm", meeting_id: str = AMI_MEETING_ID, audio_out: str | None = None):
    ds = load_dataset("edinburghcstr/ami", condition, split="validation")
    rows = [r for r in ds if r[KEY_MEETING] == meeting_id]
    if not rows:
        raise ValueError(f"meeting {meeting_id} not in AMI/{condition} validation split")
    segments = [
        Segment(
            speaker=str(r[KEY_SPEAKER]),
            start=float(r[KEY_START]),
            end=float(r[KEY_END]),
            text=str(r[KEY_TEXT]),
        )
        for r in rows
    ]
    # write the meeting audio to disk for engine adapters that take a path
    if audio_out:
        import soundfile as sf
        a = rows[0]["audio"]
        sf.write(audio_out, a["array"], a["sampling_rate"])
    return Reference(uri=f"ami_{condition}_{meeting_id}", segments=segments), audio_out
```

- [ ] **Step 3: Write smoke test**

```python
from eval.corpora.ami import load


def test_ami_loads_multispeaker_reference():
    ref, _ = load(condition="sdm")
    assert len(ref.segments) > 10
    assert len({s.speaker for s in ref.segments}) >= 2
    assert all(s.end >= s.start for s in ref.segments)
```

- [ ] **Step 4: Run smoke test**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_ami_smoke.py -v
```
Expected: PASS (downloads the slice on first run; may be slow). If audio rows are per-utterance rather than per-meeting, set `audio_out` handling to concatenate — note this in the test output and adjust Step 2.

- [ ] **Step 5: Commit**

```bash
git add eval/corpora/ami.py tests/test_ami_smoke.py && git commit -m "eval: AMI corpus loader"
```

---

## Task 9: Earnings-21 corpus loader (license-gated; introspect + smoke)

**Files:**
- Create: `eval/corpora/earnings21.py`
- Test: `tests/test_earnings21_smoke.py`

- [ ] **Step 1: Confirm license permits use (spec §6 contingency)**

```bash
~/.pyenv/versions/tathurell-eval/bin/python - <<'PY'
import urllib.request
print(urllib.request.urlopen(
  "https://raw.githubusercontent.com/revdotcom/speech-datasets/main/earnings21/LICENSE.md"
).read().decode()[:2000])
PY
```
Expected: prints the license. If it forbids this evaluation use, STOP this task, remove Earnings-21 from `run_bakeoff.py`'s corpus list, and proceed with AMI + CHiME-6 (spec §6 fallback). Otherwise continue.

- [ ] **Step 2: Introspect the Earnings-21 transcript format**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -c "
from datasets import load_dataset
ds = load_dataset('Revai/earnings21', split='test', streaming=True)
print(next(iter(ds)).keys())
"
```
Expected: prints keys (audio + transcript + speaker fields). Record exact names; pick one call id as the pinned slice. If the HF dataset lacks speaker turns, fall back to the GitHub repo's per-file `*.nlp` token tables (columns include `speaker`) + speaker-segments file, and parse those instead.

- [ ] **Step 3: Implement `eval/corpora/earnings21.py`**

```python
"""Earnings-21 loader -> Reference. Telephony/conference audio with dense proper
names. Pins one call as the slice. Field names from Task 9 Step 2 introspection.
"""
from datasets import load_dataset
from eval.corpora.base import Segment, Reference

KEY_ID = "id"
KEY_SPEAKER = "speaker"
KEY_TEXT = "text"
KEY_START = "start"
KEY_END = "end"

EARNINGS_CALL_ID = "4341191"  # set from Step 2 introspection


def load(call_id: str = EARNINGS_CALL_ID, audio_out: str | None = None):
    ds = load_dataset("Revai/earnings21", split="test")
    rows = [r for r in ds if str(r[KEY_ID]) == call_id]
    if not rows:
        raise ValueError(f"call {call_id} not found in earnings21")
    segments = [
        Segment(str(r[KEY_SPEAKER]), float(r[KEY_START]), float(r[KEY_END]), str(r[KEY_TEXT]))
        for r in rows
    ]
    if audio_out:
        import soundfile as sf
        a = rows[0]["audio"]
        sf.write(audio_out, a["array"], a["sampling_rate"])
    return Reference(uri=f"earnings21_{call_id}", segments=segments), audio_out
```

- [ ] **Step 4: Write + run smoke test**

```python
from eval.corpora.earnings21 import load


def test_earnings21_loads_multispeaker_reference():
    ref, _ = load()
    assert len(ref.segments) > 10
    assert len({s.speaker for s in ref.segments}) >= 2
```

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_earnings21_smoke.py -v
```
Expected: PASS. Adjust KEY_* / parsing per Step 2 if it fails on field names.

- [ ] **Step 5: Commit**

```bash
git add eval/corpora/earnings21.py tests/test_earnings21_smoke.py && git commit -m "eval: Earnings-21 corpus loader"
```

---

## Task 10: CHiME-6 corpus loader (introspect + smoke)

**Files:**
- Create: `eval/corpora/chime6.py`
- Test: `tests/test_chime6_smoke.py`

- [ ] **Step 1: Fetch only the transcriptions tarball + one dev session's distant audio**

```bash
mkdir -p eval/data/chime6
# transcriptions are tiny (~2.4M); audio is large — download only what the slice needs.
# From openslr #150 mirror, fetch CHiME6_transcriptions.tar.gz and ONE dev session
# (e.g. S02) distant-array wav. Record exact URLs used here:
#   curl -L -o eval/data/chime6/transcriptions.tar.gz <openslr-150 transcriptions url>
```
Expected: transcription JSONs present under `eval/data/chime6/`. Inspect one:
```bash
~/.pyenv/versions/tathurell-eval/bin/python -c "
import json, glob
f = sorted(glob.glob('eval/data/chime6/**/S02*.json', recursive=True))[0]
d = json.load(open(f)); print(type(d), d[0].keys() if isinstance(d, list) else d.keys())
"
```
Expected: prints fields per utterance (`speaker`, `start_time`, `end_time`, `words`/`ref`). Record them.

- [ ] **Step 2: Implement `eval/corpora/chime6.py`**

```python
"""CHiME-6 loader -> Reference from the per-session transcription JSON. Real
dinner-party audio (far-field arrays). Uses the distant-array channel as the
realistic condition. Field names from Task 10 Step 1 introspection.
"""
import json
from eval.corpora.base import Segment, Reference

KEY_SPEAKER = "speaker"
KEY_START = "start_time"
KEY_END = "end_time"
KEY_TEXT = "words"


def _t(v):
    # CHiME-6 times are often "H:MM:SS.ss" strings or floats.
    if isinstance(v, (int, float)):
        return float(v)
    h, m, s = v.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load(json_path: str, session: str = "S02"):
    data = json.load(open(json_path))
    segments = [
        Segment(str(u[KEY_SPEAKER]), _t(u[KEY_START]), _t(u[KEY_END]), str(u[KEY_TEXT]))
        for u in data
    ]
    return Reference(uri=f"chime6_{session}", segments=segments)
```

- [ ] **Step 3: Write + run smoke test** (point at the real downloaded JSON)

```python
import glob
from eval.corpora.chime6 import load


def test_chime6_loads_multispeaker_reference():
    path = sorted(glob.glob("eval/data/chime6/**/S02*.json", recursive=True))[0]
    ref = load(path)
    assert len(ref.segments) > 10
    assert len({s.speaker for s in ref.segments}) >= 3  # dinner parties have 4
```

```bash
~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_chime6_smoke.py -v
```
Expected: PASS. Adjust `_t`/KEY_* per Step 1 if needed.

- [ ] **Step 4: Commit**

```bash
echo "eval/data/" >> .gitignore
git add eval/corpora/chime6.py tests/test_chime6_smoke.py .gitignore && git commit -m "eval: CHiME-6 corpus loader"
```

---

## Task 11: vosk+pyannote baseline adapter (introspect + smoke)

**Files:**
- Create: `eval/engines/vosk_pyannote.py`
- Test: `tests/test_vosk_engine_smoke.py`

- [ ] **Step 1: Implement `eval/engines/vosk_pyannote.py`**

This reuses the production pipeline's logic but (a) emits per-word start/end, (b) **includes the `FinalResult()` tail** that the production script drops — required for a fair ASR comparison — and (c) uses `assign_speakers_max_overlap` instead of the old heuristic.

```python
"""Baseline engine: vosk (gigaspeech) ASR + pyannote 3.1 diarization +
max-overlap attribution. Mirrors Tathurell's production pipeline, fixed:
includes the trailing FinalResult() (production drops it) so ASR coverage is
fair, and uses max-overlap attribution.
"""
import io
import json
import os

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from pyannote.audio import Pipeline

from eval.engines.base import assign_speakers_max_overlap

VOSK_MODEL = "/Users/benmorsillo/code/ASSISTANTS/JOAN/models/vosk-model-en-us-0.42-gigaspeech"


class VoskPyannote:
    name = "vosk_pyannote"

    def __init__(self):
        self._model = Model(VOSK_MODEL)
        self._dia = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"]
        ).to(torch.device("mps"))

    def _diarize(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        dia = self._dia({"waveform": waveform, "sample_rate": sr})
        return [
            {"speaker": spk, "start": turn.start, "end": turn.end}
            for turn, _, spk in dia.itertracks(yield_label=True)
        ]

    def _asr(self, audio_path):
        rec = KaldiRecognizer(self._model, 16000)
        rec.SetWords(True)
        audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)
        pcm = np.array(audio.get_array_of_samples(), dtype=np.int16)
        words = []
        for i in range(0, len(pcm), 4000):
            if rec.AcceptWaveform(pcm[i:i + 4000].tobytes()):
                for w in json.loads(rec.Result()).get("result", []):
                    words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
        for w in json.loads(rec.FinalResult()).get("result", []):   # the dropped tail
            words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
        return words

    def transcribe(self, audio_path):
        turns = self._diarize(audio_path)
        words = self._asr(audio_path)
        return assign_speakers_max_overlap(words, turns)
```

Note: `AudioSegment.get_array_of_samples()` replaces the production script's
`np.frombuffer(wav_with_header)` bug (which read the 44-byte WAV header as audio).

- [ ] **Step 2: Write smoke test against a known short clip**

```python
import os, pytest
from eval.engines.vosk_pyannote import VoskPyannote


@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="needs HF_TOKEN")
def test_vosk_engine_runs_on_clip():
    words = VoskPyannote().transcribe("dollop_test_a.mp3")
    assert len(words) > 50
    assert all("speaker" in w for w in words)
    assert {w["speaker"] for w in words} != {None}
```

- [ ] **Step 3: Run smoke test**

```bash
HF_TOKEN=$HF_TOKEN ~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_vosk_engine_smoke.py -v
```
Expected: PASS (a few minutes; loads vosk + pyannote).

- [ ] **Step 4: Commit**

```bash
git add eval/engines/vosk_pyannote.py tests/test_vosk_engine_smoke.py && git commit -m "eval: vosk+pyannote baseline adapter"
```

---

## Task 12: WhisperX adapter (introspect + smoke)

**Files:**
- Create: `eval/engines/whisperx_stack.py`
- Test: `tests/test_whisperx_engine_smoke.py`

- [ ] **Step 1: Introspect the installed whisperx API (version-sensitive)**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -c "
import whisperx
print('top:', [x for x in dir(whisperx) if not x.startswith('_')])
try:
    from whisperx import DiarizationPipeline; print('diar: top-level')
except ImportError:
    from whisperx.diarize import DiarizationPipeline; print('diar: whisperx.diarize')
"
```
Expected: prints available functions and which import path exposes `DiarizationPipeline`. Use the working import in Step 2. Confirm `load_model`, `load_align_model`, `align`, `assign_word_speakers` are present.

- [ ] **Step 2: Implement `eval/engines/whisperx_stack.py`**

```python
"""WhisperX engine: faster-whisper (large-v3) + wav2vec2 forced alignment +
pyannote diarization + built-in word->speaker assignment. CPU on Apple Silicon
(ctranslate2 has no MPS backend) — that runtime cost is part of the eval.

Diarization import path is version-sensitive; confirmed in Task 12 Step 1.
"""
import os
import whisperx

try:
    from whisperx import DiarizationPipeline
except ImportError:
    from whisperx.diarize import DiarizationPipeline

DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # CPU-friendly; bump to float32 if accuracy needs it


class WhisperXStack:
    name = "whisperx"

    def __init__(self):
        self._model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE)
        self._diarize = DiarizationPipeline(use_auth_token=os.environ["HF_TOKEN"], device=DEVICE)

    def transcribe(self, audio_path):
        audio = whisperx.load_audio(audio_path)
        result = self._model.transcribe(audio, batch_size=8)
        align_model, meta = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
        result = whisperx.align(result["segments"], align_model, meta, audio, DEVICE)
        diar = self._diarize(audio)
        result = whisperx.assign_word_speakers(diar, result)
        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                if "start" not in w:        # alignment can drop timing on some tokens
                    continue
                words.append({
                    "word": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "speaker": w.get("speaker"),
                })
        return words
```

- [ ] **Step 3: Write smoke test**

```python
import os, pytest
from eval.engines.whisperx_stack import WhisperXStack


@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="needs HF_TOKEN")
def test_whisperx_runs_on_clip():
    words = WhisperXStack().transcribe("dollop_test_a.mp3")
    assert len(words) > 50
    assert all("speaker" in w for w in words)
```

- [ ] **Step 4: Run smoke test**

```bash
HF_TOKEN=$HF_TOKEN ~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_whisperx_engine_smoke.py -v
```
Expected: PASS (slow on CPU; downloads large-v3 + alignment model on first run). If `assign_word_speakers` arg order differs in this version, swap per Step 1 dir() output.

- [ ] **Step 5: Commit**

```bash
git add eval/engines/whisperx_stack.py tests/test_whisperx_engine_smoke.py && git commit -m "eval: whisperx adapter"
```

---

## Task 13: mlx-whisper + pyannote adapter (smoke)

**Files:**
- Create: `eval/engines/mlxwhisper_pyannote.py`
- Test: `tests/test_mlx_engine_smoke.py`

- [ ] **Step 1: Implement `eval/engines/mlxwhisper_pyannote.py`**

```python
"""mlx-whisper (Apple-Silicon-native, large-v3) ASR + pyannote 3.1 diarization +
max-overlap attribution. The speed contender on this Mac.
"""
import os
import mlx_whisper
import torch
import torchaudio
from pyannote.audio import Pipeline

from eval.engines.base import assign_speakers_max_overlap

MLX_REPO = "mlx-community/whisper-large-v3-mlx"


class MlxWhisperPyannote:
    name = "mlxwhisper_pyannote"

    def __init__(self):
        self._dia = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"]
        ).to(torch.device("mps"))

    def _diarize(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        dia = self._dia({"waveform": waveform, "sample_rate": sr})
        return [
            {"speaker": spk, "start": turn.start, "end": turn.end}
            for turn, _, spk in dia.itertracks(yield_label=True)
        ]

    def transcribe(self, audio_path):
        out = mlx_whisper.transcribe(
            audio_path, path_or_hf_repo=MLX_REPO, word_timestamps=True
        )
        words = []
        for seg in out["segments"]:
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })
        return assign_speakers_max_overlap(words, self._diarize(audio_path))
```

- [ ] **Step 2: Write + run smoke test**

```python
import os, pytest
from eval.engines.mlxwhisper_pyannote import MlxWhisperPyannote


@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="needs HF_TOKEN")
def test_mlx_runs_on_clip():
    words = MlxWhisperPyannote().transcribe("dollop_test_a.mp3")
    assert len(words) > 50
    assert all("speaker" in w for w in words)
```

```bash
HF_TOKEN=$HF_TOKEN ~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_mlx_engine_smoke.py -v
```
Expected: PASS. If mlx-whisper word dicts use different keys, print `out["segments"][0]["words"][0]` and adjust.

- [ ] **Step 3: Commit**

```bash
git add eval/engines/mlxwhisper_pyannote.py tests/test_mlx_engine_smoke.py && git commit -m "eval: mlx-whisper+pyannote adapter"
```

---

## Task 14: Orchestrator + results table + README

**Files:**
- Create: `eval/run_bakeoff.py`
- Create: `eval/README.md`
- Test: `tests/test_run_bakeoff_smoke.py`

- [ ] **Step 1: Implement `eval/run_bakeoff.py`**

```python
"""Run engines x corpus slices, compute WER/cpWER/DER + runtime/memory, emit a
markdown results table and save each engine's transcript for inspection.

Usage: HF_TOKEN=... python -m eval.run_bakeoff
"""
import resource
import time
from collections import defaultdict

from eval.engines.vosk_pyannote import VoskPyannote
from eval.engines.whisperx_stack import WhisperXStack
from eval.engines.mlxwhisper_pyannote import MlxWhisperPyannote
from eval.corpora import ami, earnings21, chime6  # noqa: F401 (chime6 used via SLICES)
from eval.metrics.wer import word_error_rate
from eval.metrics.cpwer import cp_wer
from eval.metrics.der import diarization_error_rate
from eval.corpora.base import Reference


def words_to_hyp(words):
    """Group engine words by predicted speaker -> {speaker: text} and flat text."""
    by_spk = defaultdict(list)
    for w in words:
        by_spk[w["speaker"]].append(w["word"])
    per_spk = {spk: " ".join(ws) for spk, ws in by_spk.items()}
    flat = " ".join(w["word"] for w in words)
    return per_spk, flat


def words_to_rttm(words, uri):
    """Collapse consecutive same-speaker words into turns -> RTTM text."""
    segs = []
    for w in words:
        if segs and segs[-1]["speaker"] == w["speaker"]:
            segs[-1]["end"] = w["end"]
        else:
            segs.append({"speaker": str(w["speaker"]), "start": w["start"], "end": w["end"]})
    lines = [
        f"SPEAKER {uri} 1 {s['start']:.3f} {s['end'] - s['start']:.3f} <NA> <NA> {s['speaker']} <NA> <NA>"
        for s in segs
    ]
    return "\n".join(lines) + "\n"


def run():
    # (reference_loader_thunk, audio_path) per slice. audio_path written by loaders.
    slices = []
    ami_ref, ami_audio = ami.load(condition="sdm", audio_out="eval/data/ami_sdm.wav")
    slices.append(("ami", ami_ref, ami_audio))
    # earnings21 / chime6 appended here once their audio paths are wired (Tasks 9/10).

    engines = [VoskPyannote(), WhisperXStack(), MlxWhisperPyannote()]
    rows = []
    for corpus_name, ref, audio in slices:
        ref_rttm = f"eval/data/{ref.uri}.ref.rttm"
        open(ref_rttm, "w").write(ref.to_rttm())
        ref_spk_text = ref.per_speaker_text()
        ref_flat = " ".join(s.text for s in sorted(ref.segments, key=lambda s: s.start))
        for eng in engines:
            t0 = time.time()
            words = eng.transcribe(audio)
            dt = time.time() - t0
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
            per_spk, flat = words_to_hyp(words)
            open(f"eval/data/{ref.uri}.{eng.name}.txt", "w").write(
                "\n".join(f"{s}: {t}" for s, t in per_spk.items())
            )
            hyp_rttm = f"eval/data/{ref.uri}.{eng.name}.rttm"
            open(hyp_rttm, "w").write(words_to_rttm(words, ref.uri))
            rows.append({
                "corpus": corpus_name, "engine": eng.name,
                "WER": word_error_rate(ref_flat, flat),
                "cpWER": cp_wer(reference=ref_spk_text, hypothesis=per_spk),
                "DER": diarization_error_rate(ref_rttm, hyp_rttm),
                "sec": round(dt, 1), "mem_MB": round(mem_mb, 0),
            })

    cols = ["corpus", "engine", "WER", "cpWER", "DER", "sec", "mem_MB"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(
            f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols
        ) + " |")
    return rows


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Write a smoke test (one engine, AMI slice only, asserts a finite table row)**

```python
import os, pytest


@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="needs HF_TOKEN")
def test_bakeoff_produces_metric_rows(monkeypatch):
    import eval.run_bakeoff as rb
    from eval.engines.vosk_pyannote import VoskPyannote
    monkeypatch.setattr(rb, "WhisperXStack", lambda: VoskPyannote())   # skip heavy engines
    monkeypatch.setattr(rb, "MlxWhisperPyannote", lambda: VoskPyannote())
    rows = rb.run()
    assert rows and all(0.0 <= r["cpWER"] for r in rows)
    assert all(r["WER"] == r["WER"] for r in rows)  # not NaN
```

- [ ] **Step 3: Run smoke test**

```bash
HF_TOKEN=$HF_TOKEN ~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_run_bakeoff_smoke.py -v
```
Expected: PASS — prints a table with finite WER/cpWER/DER for the AMI slice.

- [ ] **Step 4: Write `eval/README.md`**

```markdown
# Engine bake-off

Isolated accuracy evaluation for Tathurell's transcription engine choice.
See `docs/superpowers/specs/2026-06-13-engine-bakeoff-design.md`.

## Run
    pyenv activate tathurell-eval        # separate from the project's tathurell-reqs
    export HF_TOKEN=<a fresh huggingface token, model license accepted>
    python -m eval.run_bakeoff

Outputs a markdown table (WER / cpWER / DER / runtime / memory) and per-engine
transcripts under `eval/data/`. Headline metric is **cpWER** on the realistic
far-field/telephony condition; runtime is the practicality tiebreaker.

## Add an engine
Implement `eval/engines/<name>.py` exposing `transcribe(audio_path) -> [Word]`
(`Word = {word, start, end, speaker}`) and add it to `run_bakeoff.engines`.

## Add a corpus
Implement `eval/corpora/<name>.py` returning a `Reference` and add it to
`run_bakeoff.slices`.
```

- [ ] **Step 5: Run the full bake-off and capture the table into the recommendation memo**

```bash
HF_TOKEN=$HF_TOKEN ~/.pyenv/versions/tathurell-eval/bin/python -m eval.run_bakeoff | tee docs/superpowers/specs/2026-06-13-bakeoff-results.md
```
Expected: a full table across all wired corpora and engines. Then add a one-paragraph recommendation (winner by mean cpWER on realistic conditions, runtime tiebreaker) to that results file.

- [ ] **Step 6: Commit**

```bash
git add eval/run_bakeoff.py eval/README.md tests/test_run_bakeoff_smoke.py docs/superpowers/specs/2026-06-13-bakeoff-results.md && git commit -m "eval: bake-off orchestrator, README, and results"
```

---

## Self-review

**Spec coverage:**
- 3 engine stacks → Tasks 11, 12, 13. ✓
- 3 metrics (cpWER headline, WER, DER) + runtime/memory → Tasks 5, 6, 7; runtime/memory in Task 14. ✓
- Shared normalizer fairness → Task 4, used by Tasks 5/6. ✓
- 3 corpora, dev slices, common reference format → Tasks 2, 8, 9, 10. ✓
- Far-field-primary condition → Task 8 (sdm), Task 10 (distant), Earnings as-is. ✓
- Separate venv isolation → Task 1. ✓
- Max-overlap attribution → Task 3, used by baseline + mlx. ✓
- Earnings-21 license contingency → Task 9 Step 1 (with fallback to drop it). ✓
- Decision rule (mean cpWER, runtime tiebreaker) → Task 14 Step 5 + README. ✓
- Out-of-scope items (pipeline rewrite, packaging, naming UX, transcoding) → not in plan. ✓ (The production `FinalResult` tail is fixed *inside the eval baseline adapter* only, for fair comparison — Task 11; the production script itself stays out of scope per spec §8.)

**Placeholder scan:** No "TBD"/"implement later". The introspect-first steps (Tasks 8–13) give exact commands + expected output + the constants to set from them; this is verification, not a placeholder — it exists because corpus formats and whisperx versions can't be pinned without the installed libraries, and inventing field names would be worse.

**Type consistency:** `Word = {word,start,end,speaker}` and `Turn = {speaker,start,end}` used consistently across base.py, all adapters, and run_bakeoff. `Reference.per_speaker_text()` / `Reference.to_rttm()` names match their call sites in Task 14. `cp_wer(reference=, hypothesis=)` signature matches Task 6 definition and Task 14 call.
