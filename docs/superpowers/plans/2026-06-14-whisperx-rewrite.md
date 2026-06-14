# WhisperX Pipeline Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Tathurell's vosk+pyannote+attribution transcription with WhisperX, behind a small owned `tathurell/` package and a thin CLI, keeping the interactive speaker-naming flow and `<audio>.transcription.txt` output.

**Architecture:** A pure `tathurell/naming.py` (grouping + name application), a `tathurell/whisperx_core.py` owning `WhisperXTranscriber` (promoted from the bake-off's `eval/engines/whisperx_stack.py`), the eval adapter refactored to re-use it (DRY), and a rewritten `tathurell_transcribe.py` CLI.

**Tech Stack:** Python 3.10 in the `tathurell-eval` venv, whisperx 3.8.6 (large-v3, CPU), pyannote.audio 4.0.4, pytest.

Spec: `docs/superpowers/specs/2026-06-14-whisperx-rewrite-design.md`. All commands use the eval venv python: `~/.pyenv/versions/tathurell-eval/bin/python`.

---

## File structure

```
tathurell/
  __init__.py
  naming.py          # PURE: group_by_speaker, apply_names
  whisperx_core.py   # resolve_hf_token + WhisperXTranscriber (owns WhisperX logic)
tathurell_transcribe.py        # rewritten thin CLI (repo root, imports tathurell.*)
eval/engines/whisperx_stack.py # refactored: thin wrapper over WhisperXTranscriber
tests/
  test_naming.py             # PURE unit tests
  test_hf_token.py           # PURE unit tests (env/file/exit)
  test_cli_naming.py         # PURE: prompt_names EOF fallback
  test_whisperx_core_smoke.py# token-gated smoke
.python-version                # tathurell-reqs -> tathurell-eval
```

---

## Task 1: `tathurell` package + pure naming logic

**Files:**
- Create: `tathurell/__init__.py` (empty), `tathurell/naming.py`
- Test: `tests/test_naming.py`

- [ ] **Step 1: Create the empty package file**

Create `tathurell/__init__.py` as an empty file.

- [ ] **Step 2: Write the failing test** (`tests/test_naming.py`)

```python
from tathurell.naming import group_by_speaker, apply_names


def test_group_starts_new_run_with_triggering_word():
    words = [
        {"word": "hello", "speaker": "A"},
        {"word": "there", "speaker": "A"},
        {"word": "hi", "speaker": "B"},
        {"word": "again", "speaker": "A"},
    ]
    assert group_by_speaker(words) == [
        {"speaker": "A", "text": "hello there"},
        {"speaker": "B", "text": "hi"},
        {"speaker": "A", "text": "again"},
    ]


def test_group_empty_input():
    assert group_by_speaker([]) == []


def test_apply_names_formats_and_falls_back_to_label():
    groups = [{"speaker": "A", "text": "hello"}, {"speaker": "B", "text": "hi"}]
    assert apply_names(groups, {"A": "dave"}) == "dave: hello\nB: hi"
```

- [ ] **Step 3: Run test, verify it fails**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_naming.py -v`
Expected: FAIL (ImportError: cannot import name 'group_by_speaker').

- [ ] **Step 4: Implement `tathurell/naming.py`**

```python
"""Pure grouping + naming for diarized word lists. No models, no I/O.

A word is {"word": str, "speaker": str | None, ...}. group_by_speaker collapses
consecutive same-speaker words into runs; the word that triggers a speaker change
starts the new run (the bug the original code had: it dropped that word).
"""


def group_by_speaker(words):
    """Collapse consecutive same-speaker words into [{"speaker", "text"}] runs."""
    groups = []
    for w in words:
        spk = w["speaker"]
        if groups and groups[-1]["speaker"] == spk:
            groups[-1]["text"] += f" {w['word']}"
        else:
            groups.append({"speaker": spk, "text": w["word"]})
    return groups


def apply_names(groups, names):
    """Render runs as "{name}: {text}" lines. Unmapped speakers use their label."""
    return "\n".join(
        f"{names.get(g['speaker'], g['speaker'])}: {g['text']}" for g in groups
    )
```

- [ ] **Step 5: Run test, verify it passes**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_naming.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add tathurell/__init__.py tathurell/naming.py tests/test_naming.py
git commit -m "tathurell: pure speaker grouping + name application"
```

---

## Task 2: token resolution + WhisperXTranscriber

**Files:**
- Create: `tathurell/whisperx_core.py`
- Test: `tests/test_hf_token.py`, `tests/test_whisperx_core_smoke.py`

- [ ] **Step 1: Write the failing test for token resolution** (`tests/test_hf_token.py`)

```python
import pytest
from tathurell.whisperx_core import resolve_hf_token


def test_env_token_wins(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert resolve_hf_token(token_file="/nonexistent") == "hf_env"


def test_file_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    f = tmp_path / ".hf_token"
    f.write_text("hf_file\n")
    assert resolve_hf_token(token_file=str(f)) == "hf_file"


def test_missing_exits(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        resolve_hf_token(token_file=str(tmp_path / "absent"))
```

- [ ] **Step 2: Run test, verify it fails**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_hf_token.py -v`
Expected: FAIL (ImportError: cannot import name 'resolve_hf_token').

- [ ] **Step 3: Implement `tathurell/whisperx_core.py`**

```python
"""WhisperX transcription engine, owned by the production tool.

Promoted from the bake-off adapter. Version facts (whisperx 3.8.6 + pyannote 4.0.4),
all verified during the bake-off and required for correctness:
  - DiarizationPipeline lives in whisperx.diarize (not top-level whisperx).
  - pyannote 4.x auth arg is `token=` (not `use_auth_token=`).
  - faster-whisper (ctranslate2) has no MPS backend -> device must be "cpu".
  - aligned word dicts have keys word/start/end (+ score); start/end absent on
    tokens alignment could not pin -> skip those.
"""
import os
from pathlib import Path

import whisperx

try:
    from whisperx import DiarizationPipeline
except ImportError:
    from whisperx.diarize import DiarizationPipeline

DEFAULT_TOKEN_FILE = "eval/data/.hf_token"


def resolve_hf_token(token_file: str = DEFAULT_TOKEN_FILE) -> str:
    """HF token from $HF_TOKEN, else from token_file, else exit with guidance."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    p = Path(token_file)
    if p.exists():
        tok = p.read_text().strip()
        if tok:
            return tok
    raise SystemExit(
        "HF_TOKEN is not set. Create a token at https://huggingface.co/settings/tokens, "
        "accept the gates for pyannote/speaker-diarization-3.1, pyannote/segmentation-3.0, "
        "and pyannote/speaker-diarization-community-1, then either export HF_TOKEN=... or "
        f"write it to {token_file}."
    )


class WhisperXTranscriber:
    """Load WhisperX (large-v3, CPU) + pyannote diarization once; transcribe to words."""

    def __init__(self, model="large-v3", device="cpu", compute_type="int8", token=None):
        self._device = device
        self._model = whisperx.load_model(model, device, compute_type=compute_type)
        self._diarize = DiarizationPipeline(
            token=token or resolve_hf_token(), device=device
        )

    def transcribe(self, audio_path: str) -> list:
        """Return [{"word", "start", "end", "speaker"}] for the audio file."""
        audio = whisperx.load_audio(audio_path)
        result = self._model.transcribe(audio, batch_size=8)
        align_model, meta = whisperx.load_align_model(
            language_code=result["language"], device=self._device
        )
        result = whisperx.align(result["segments"], align_model, meta, audio, self._device)
        diar = self._diarize(audio)
        result = whisperx.assign_word_speakers(diar, result)
        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                if "start" not in w:  # alignment dropped timing for this token
                    continue
                words.append({
                    "word": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "speaker": w.get("speaker"),
                })
        return words
```

- [ ] **Step 4: Run token test, verify it passes**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_hf_token.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify the module imports WITHOUT a token** (lazy: token only needed at instantiation)

Run: `env -u HF_TOKEN ~/.pyenv/versions/tathurell-eval/bin/python -c "import tathurell.whisperx_core; print('import ok')"`
Expected: prints `import ok` (no SystemExit — resolve_hf_token is only called when WhisperXTranscriber is constructed).

- [ ] **Step 6: Write the token-gated smoke test** (`tests/test_whisperx_core_smoke.py`)

```python
import os
import pytest
from tathurell.whisperx_core import WhisperXTranscriber


@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="needs HF_TOKEN (pyannote)")
def test_transcriber_runs_on_clip():
    words = WhisperXTranscriber().transcribe("dollop_test_a.mp3")
    assert len(words) > 50
    assert all({"word", "start", "end", "speaker"} <= set(w) for w in words)
```

- [ ] **Step 7: Run the smoke test with the token**

Run: `HF_TOKEN=$(tr -d '[:space:]' < eval/data/.hf_token) ~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_whisperx_core_smoke.py -v`
Expected: PASS (downloads large-v3 if not cached; CPU, a few min). Without a token it SKIPS.

- [ ] **Step 8: Commit**

```bash
git add tathurell/whisperx_core.py tests/test_hf_token.py tests/test_whisperx_core_smoke.py
git commit -m "tathurell: WhisperXTranscriber + HF token resolution"
```

---

## Task 3: refactor the eval adapter to reuse the core (DRY)

**Files:**
- Modify: `eval/engines/whisperx_stack.py` (replace its body with a thin wrapper)
- Test: `tests/test_whisperx_engine_smoke.py` (existing — must still pass/skip)

- [ ] **Step 1: Replace `eval/engines/whisperx_stack.py` entirely**

```python
"""Bake-off adapter: thin wrapper over the production WhisperXTranscriber so the
bake-off and the production tool share ONE WhisperX implementation (DRY).

Importing this module must NOT require a token (WhisperXTranscriber resolves the
token only at construction, not import).
"""
from tathurell.whisperx_core import WhisperXTranscriber


class WhisperXStack:
    name = "whisperx"

    def __init__(self):
        self._t = WhisperXTranscriber()

    def transcribe(self, audio_path):
        return self._t.transcribe(audio_path)
```

- [ ] **Step 2: Verify token-free import still holds**

Run: `env -u HF_TOKEN ~/.pyenv/versions/tathurell-eval/bin/python -c "import eval.engines.whisperx_stack; print('import ok')"`
Expected: prints `import ok`.

- [ ] **Step 3: Verify the existing eval smoke still skips without a token**

Run: `env -u HF_TOKEN ~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_whisperx_engine_smoke.py -v`
Expected: 1 skipped (reason: needs HF_TOKEN).

- [ ] **Step 4: Verify it still RUNS with a token** (proves the refactor preserved behavior)

Run: `HF_TOKEN=$(tr -d '[:space:]' < eval/data/.hf_token) ~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_whisperx_engine_smoke.py -v`
Expected: PASS (CPU, a few min).

- [ ] **Step 5: Commit**

```bash
git add eval/engines/whisperx_stack.py
git commit -m "eval: whisperx adapter reuses tathurell.WhisperXTranscriber (DRY)"
```

---

## Task 4: rewrite the CLI + switch the project env

**Files:**
- Overwrite: `tathurell_transcribe.py`
- Modify: `.python-version`
- Test: `tests/test_cli_naming.py`

- [ ] **Step 1: Write the failing test for the naming prompt's EOF fallback** (`tests/test_cli_naming.py`)

```python
import builtins
import importlib


def test_prompt_names_falls_back_to_label_on_eof(monkeypatch):
    cli = importlib.import_module("tathurell_transcribe")

    def raise_eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", raise_eof)
    groups = [
        {"speaker": "A", "text": "hello"},
        {"speaker": "B", "text": "hi"},
        {"speaker": "A", "text": "again"},  # repeat speaker -> asked once
    ]
    names = cli.prompt_names(groups)
    assert names == {"A": "A", "B": "B"}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_cli_naming.py -v`
Expected: FAIL (ModuleNotFoundError or AttributeError: prompt_names — the current script has no such function and runs on import).

- [ ] **Step 3: Overwrite `tathurell_transcribe.py`**

```python
#!/usr/bin/env python3
"""Transcribe + diarize an audio file with WhisperX, name speakers, write text.

Output: "<name>: <text>" lines, one per speaker run, to <audio>.transcription.txt
(or --output). Runs in the tathurell-eval venv. Needs HF_TOKEN (see whisperx_core).
"""
import argparse
import sys

from tathurell.naming import apply_names, group_by_speaker
from tathurell.whisperx_core import WhisperXTranscriber, resolve_hf_token


def prompt_names(groups):
    """Ask the user to name each distinct speaker (once). EOF -> use the label."""
    names = {}
    for g in groups:
        spk = g["speaker"]
        if spk in names:
            continue
        print(f"Who said this?\n{g['text']}")
        try:
            answer = input("name: ").strip()
        except EOFError:
            answer = ""
        names[spk] = answer or str(spk)
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description="Transcribe + diarize audio with WhisperX.")
    ap.add_argument("audio_path")
    ap.add_argument("--output", default=None,
                    help="output path (default: <audio_path>.transcription.txt)")
    ap.add_argument("--model", default="large-v3", help="Whisper model (default: large-v3)")
    args = ap.parse_args(argv)

    token = resolve_hf_token()  # clean early exit before loading models if missing
    words = WhisperXTranscriber(model=args.model, token=token).transcribe(args.audio_path)
    if not words:
        print("[tathurell] WARNING: no words transcribed.", file=sys.stderr)

    groups = group_by_speaker(words)
    names = prompt_names(groups)
    text = apply_names(groups, names)

    out_path = args.output or f"{args.audio_path}.transcription.txt"
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[tathurell] wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the EOF test, verify it passes**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_cli_naming.py -v`
Expected: PASS. (Importing `tathurell_transcribe` no longer executes the pipeline — it's guarded by `__main__` — so the import succeeds.)

- [ ] **Step 5: Verify the argument parser works (no models)**

Run: `~/.pyenv/versions/tathurell-eval/bin/python tathurell_transcribe.py --help`
Expected: prints usage with `audio_path`, `--output`, `--model`.

- [ ] **Step 6: Switch the project env**

Overwrite `.python-version` so its only contents are:
```
tathurell-eval
```

- [ ] **Step 7: End-to-end smoke with the token (auto-answer the name prompts)**

Run:
```bash
HF_TOKEN=$(tr -d '[:space:]' < eval/data/.hf_token) bash -c \
  'yes "spk" | ~/.pyenv/versions/tathurell-eval/bin/python tathurell_transcribe.py dollop_test_a.mp3 --output /tmp/tath_smoke.txt'
test -s /tmp/tath_smoke.txt && head -3 /tmp/tath_smoke.txt
```
Expected: writes a non-empty `/tmp/tath_smoke.txt` whose lines look like `spk: <words>` (the `yes "spk"` feeds the interactive name prompts). A few minutes on CPU.

- [ ] **Step 8: Run the full repo test suite (no token; gated tests skip)**

Run: `env -u HF_TOKEN ~/.pyenv/versions/tathurell-eval/bin/python -m pytest -q`
Expected: all pure tests pass (naming, hf_token, cli_naming, plus the existing eval suite); engine smokes skip.

- [ ] **Step 9: Commit**

```bash
git add tathurell_transcribe.py .python-version tests/test_cli_naming.py
git commit -m "tathurell: rewrite CLI on WhisperX; switch project env to tathurell-eval"
```

---

## Self-Review

**Spec coverage:**
- §3 `whisperx_core.py` WhisperXTranscriber → Task 2. ✓
- §3 `naming.py` group_by_speaker/apply_names → Task 1. ✓
- §3 thin CLI (args, token, transcribe, group, name, write) → Task 4. ✓
- §3 eval adapter re-uses the core (DRY) → Task 3. ✓
- §2 env = tathurell-eval, `.python-version` switched → Task 4 Step 6. ✓
- §5 config: HF_TOKEN env → file fallback → exit; CLI `--output`/`--model` → Task 2 (resolve_hf_token) + Task 4 (argparse). ✓
- §6 error handling: missing token exit (Task 2/4), zero-words warn (Task 4 Step 3), EOF naming fallback (Task 4 Step 1/3). ✓
- §7 testing: pure unit tests for naming + token + prompt_names; token-gated core smoke; suite still passes → Tasks 1,2,4. ✓
- §8 removals (vosk, chunk loop, attribution, FinalResult, hardcoded paths) → Task 4 overwrites the script entirely; no vosk/pyannote-direct code remains. ✓
- §9 output format unchanged (`"{name}: {text}"`, `<audio>.transcription.txt`) → Task 1 apply_names + Task 4 default output. ✓

**Placeholder scan:** none — every code/command step is concrete.

**Type consistency:** `Word = {"word","start","end","speaker"}` is produced by `WhisperXTranscriber.transcribe` (Task 2) and consumed by `group_by_speaker` (Task 1, uses `word`/`speaker`); `group_by_speaker` returns `[{"speaker","text"}]` consumed by `apply_names` and `prompt_names` (Tasks 1, 4). `resolve_hf_token(token_file=...)` signature matches its callers in Tasks 2 and 4. `WhisperXTranscriber(model=, device=, compute_type=, token=)` matches the Task 4 call `WhisperXTranscriber(model=args.model, token=token)`.
