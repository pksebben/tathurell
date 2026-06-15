# Bundled ffmpeg ("built-in transcoding") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dependency on a system-installed ffmpeg by bundling a static ffmpeg via `imageio-ffmpeg` and pointing both decode consumers (`whisperx.load_audio` and `extract_clip`) at it.

**Architecture:** A new `tathurell/ffmpeg.py` resolves the bundled binary and shims a bare `ffmpeg` onto PATH (whisperx hardcodes `subprocess.run(["ffmpeg", ...])`; PATH is the only injection point, and the bundled binary is named `ffmpeg-<plat>-<ver>`, so a symlink named `ffmpeg` is required). `extract_clip` is rewritten to call the bundled binary directly, dropping pydub (which needs ffprobe that `imageio-ffmpeg` doesn't ship). ffmpeg's decode is kept verbatim → no transcription-fidelity risk.

**Tech Stack:** Python 3.10 in the `tathurell-eval` venv, `imageio-ffmpeg` (static ffmpeg in-wheel), stdlib `subprocess`/`shutil`/`os`, pytest. All commands use `~/.pyenv/versions/tathurell-eval/bin/python` (aliased `PY` below). Spec: `docs/superpowers/specs/2026-06-15-bundled-ffmpeg-design.md`.

**File structure:**
- Create `tathurell/ffmpeg.py` — `ffmpeg_exe()` (memoized bundled path) + `ensure_ffmpeg_on_path()` (symlink shim + PATH prepend).
- Create `tests/test_ffmpeg.py` — unit + clean-PATH integration tests.
- Modify `tathurell/sampling.py` — rewrite `extract_clip` to call bundled ffmpeg directly; drop pydub import.
- Modify `tathurell/whisperx_core.py` — call `ensure_ffmpeg_on_path()` before `load_audio`.
- Modify `tests/test_sampling_clip.py` — add a clean-PATH parity test.
- Modify `requirements.txt` — add `imageio-ffmpeg`, remove `pydub`.
- Modify `eval/requirements-eval.txt` — add `pydub` (eval-only; `eval/engines/vosk_pyannote.py` still imports it).

> Throughout: `PY=~/.pyenv/versions/tathurell-eval/bin/python`. Run pytest as `env -u HF_TOKEN $PY -m pytest` from the repo root (where `dollop_test_a.mp3` lives).

---

### Task 1: Bundle imageio-ffmpeg (install + pin)

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Install imageio-ffmpeg and capture the version**

Run:
```bash
PY=~/.pyenv/versions/tathurell-eval/bin/python
$PY -m pip install imageio-ffmpeg
$PY -c "import importlib.metadata as m; print(m.version('imageio-ffmpeg'))"
```
Note the printed version (call it `X.Y.Z`) for the next step.

- [ ] **Step 2: Confirm the bundled binary exists and note its basename**

Run:
```bash
$PY -c "import imageio_ffmpeg, os; e=imageio_ffmpeg.get_ffmpeg_exe(); print(e, os.path.isfile(e), os.access(e, os.X_OK))"
```
Expected: a path ending in something like `ffmpeg-osx-arm64-v7.1`, then `True True`. The basename is **not** plain `ffmpeg` — this is exactly why Task 2 needs a symlink shim.

- [ ] **Step 3: Add the pinned dep to `requirements.txt`**

Add this line to `requirements.txt` immediately after the `Flask==3.1.3` line (use the actual `X.Y.Z` from Step 1):
```
imageio-ffmpeg==X.Y.Z
```
Leave `pydub==0.25.1` in place for now (still used by the current `extract_clip` until Task 3).

- [ ] **Step 4: Verify the runtime requirements still resolve against the venv**

Run:
```bash
for spec in whisperx==3.8.6 pyannote.audio==4.0.4 imageio-ffmpeg==X.Y.Z; do
  pkg="${spec%%==*}"; want="${spec##*==}";
  got=$($PY -m pip show "$pkg" 2>/dev/null | awk '/^Version:/{print $2}');
  [ "$got" = "$want" ] && echo "OK $pkg $got" || echo "MISMATCH $pkg want=$want got=$got";
done
```
Expected: all `OK`.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "packaging: bundle ffmpeg via imageio-ffmpeg (pinned)"
```

---

### Task 2: `tathurell/ffmpeg.py` — resolve + shim the bundled binary

**Files:**
- Create: `tathurell/ffmpeg.py`
- Test: `tests/test_ffmpeg.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ffmpeg.py`:
```python
import os
import shutil

from tathurell.ffmpeg import ffmpeg_exe, ensure_ffmpeg_on_path

SCRUBBED = "/nonexistent-tathurell-dir"  # a PATH with no system ffmpeg


def test_ffmpeg_exe_returns_executable():
    exe = ffmpeg_exe()
    assert os.path.isfile(exe)
    assert os.access(exe, os.X_OK)


def test_ensure_puts_bundled_ffmpeg_on_path(monkeypatch):
    # With PATH scrubbed of any system ffmpeg, a bare `ffmpeg` must resolve to
    # the bundled binary after ensure_ffmpeg_on_path() installs the shim.
    monkeypatch.setenv("PATH", SCRUBBED)
    assert shutil.which("ffmpeg") is None
    ensure_ffmpeg_on_path()
    found = shutil.which("ffmpeg")
    assert found is not None
    assert os.path.realpath(found) == os.path.realpath(ffmpeg_exe())


def test_ensure_is_idempotent(monkeypatch):
    # Calling it twice must not keep growing PATH.
    monkeypatch.setenv("PATH", SCRUBBED)
    ensure_ffmpeg_on_path()
    after_first = os.environ["PATH"]
    ensure_ffmpeg_on_path()
    assert os.environ["PATH"] == after_first
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_ffmpeg.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tathurell.ffmpeg'`.

- [ ] **Step 3: Implement `tathurell/ffmpeg.py`**

Create `tathurell/ffmpeg.py`:
```python
"""Resolve and activate the bundled ffmpeg, so the tool needs no system ffmpeg.

whisperx.load_audio() and tathurell.sampling.extract_clip both decode via ffmpeg.
imageio-ffmpeg ships a static ffmpeg inside its wheel; this module exposes that
binary and makes a bare `ffmpeg` on PATH resolve to it (whisperx hardcodes a
`subprocess.run(["ffmpeg", ...])` we cannot patch).
"""
import os
import shutil
import tempfile

import imageio_ffmpeg

_exe = None  # memoized bundled-binary path


def ffmpeg_exe():
    """Absolute path to the bundled ffmpeg binary (memoized)."""
    global _exe
    if _exe is None:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if not (exe and os.path.isfile(exe) and os.access(exe, os.X_OK)):
            raise RuntimeError(
                f"bundled ffmpeg missing or not executable at {exe!r}; "
                "reinstall imageio-ffmpeg"
            )
        _exe = exe
    return _exe


def ensure_ffmpeg_on_path():
    """Make a bare `ffmpeg` resolve to the bundled binary. Idempotent.

    imageio-ffmpeg's binary is named like `ffmpeg-osx-arm64-v7.1`, not `ffmpeg`,
    so putting its directory on PATH is not enough for whisperx's hardcoded
    `subprocess.run(["ffmpeg", ...])`. We keep a stable shim dir holding a symlink
    named `ffmpeg` -> the bundled binary, and prepend it to PATH.
    """
    exe = ffmpeg_exe()
    current = shutil.which("ffmpeg")
    if current and os.path.realpath(current) == os.path.realpath(exe):
        return  # a bare `ffmpeg` already resolves to the bundled binary

    shim_dir = os.path.join(tempfile.gettempdir(), "tathurell_ffmpeg")
    os.makedirs(shim_dir, exist_ok=True)
    link = os.path.join(shim_dir, "ffmpeg")
    if not (os.path.islink(link) and os.path.realpath(link) == os.path.realpath(exe)):
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(exe, link)  # POSIX; Windows would copy instead (out of scope)

    parts = os.environ.get("PATH", "").split(os.pathsep)
    if shim_dir not in parts:
        os.environ["PATH"] = shim_dir + os.pathsep + os.environ.get("PATH", "")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_ffmpeg.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tathurell/ffmpeg.py tests/test_ffmpeg.py
git commit -m "tathurell: ffmpeg.py -- resolve + PATH-shim the bundled ffmpeg"
```

---

### Task 3: Rewrite `extract_clip` to use the bundled ffmpeg directly (drop pydub)

**Files:**
- Modify: `tathurell/sampling.py` (`extract_clip`, lines ~41-51)
- Test: `tests/test_sampling_clip.py`

- [ ] **Step 1: Write the failing clean-PATH test**

Add to `tests/test_sampling_clip.py`:
```python
def test_extract_clip_works_without_system_ffmpeg(tmp_path, monkeypatch):
    # extract_clip must not depend on a system ffmpeg: scrub PATH and confirm it
    # still produces a valid ~3s clip via the bundled binary.
    monkeypatch.setenv("PATH", "/nonexistent-tathurell-dir")
    out = tmp_path / "clip.wav"
    extract_clip("dollop_test_a.mp3", 1.0, 4.0, str(out))
    with wave.open(str(out)) as w:
        dur = w.getnframes() / w.getframerate()
    assert 2.7 < dur < 3.3
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_sampling_clip.py::test_extract_clip_works_without_system_ffmpeg -v`
Expected: FAIL — current pydub-based `extract_clip` can't find a system ffmpeg/ffprobe with PATH scrubbed (FileNotFoundError / pydub `CouldntDecodeError`).

- [ ] **Step 3: Rewrite `extract_clip`**

In `tathurell/sampling.py`, replace the entire `extract_clip` function (the `from pydub import AudioSegment` block) with:
```python
def extract_clip(audio_path, start, end, out_path):
    """Write the [start, end] second slice of audio_path to out_path as a WAV.

    Calls the bundled ffmpeg directly (tathurell.ffmpeg) so no system ffmpeg is
    needed. Output is 44.1 kHz stereo 16-bit PCM, which the naming modal plays
    in-browser. `-ss` before `-i` is a fast seek (cheap on long files); `-vn`
    drops any video/cover-art stream.
    """
    import subprocess

    from tathurell.ffmpeg import ffmpeg_exe

    cmd = [
        ffmpeg_exe(), "-nostdin", "-y",
        "-ss", str(start), "-i", audio_path, "-t", str(end - start),
        "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract clip from {audio_path!r}: "
            f"{proc.stderr.decode(errors='replace')}"
        )
```

- [ ] **Step 4: Run both the new test and the existing parity test**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_sampling_clip.py -v`
Expected: PASS — both `test_extract_clip_writes_wav_of_expected_duration` (parity, unchanged) and `test_extract_clip_works_without_system_ffmpeg`.

- [ ] **Step 5: Commit**

```bash
git add tathurell/sampling.py tests/test_sampling_clip.py
git commit -m "tathurell: extract_clip uses bundled ffmpeg directly (drop pydub)"
```

---

### Task 4: Wire `ensure_ffmpeg_on_path()` into the transcribe path

**Files:**
- Modify: `tathurell/whisperx_core.py` (`WhisperXTranscriber.transcribe`, ~line 59)
- Test: `tests/test_ffmpeg.py`

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_ffmpeg.py`:
```python
def test_whisperx_load_audio_uses_bundled_ffmpeg(monkeypatch):
    # whisperx.load_audio runs a bare `ffmpeg` subprocess. With no system ffmpeg
    # on PATH, it must still decode after ensure_ffmpeg_on_path() shims it.
    import whisperx

    monkeypatch.setenv("PATH", SCRUBBED)
    ensure_ffmpeg_on_path()
    audio = whisperx.load_audio("dollop_test_a.mp3")
    assert audio is not None and len(audio) > 16000  # > 1s at 16 kHz
```

- [ ] **Step 2: Run it to verify the mechanism works**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_ffmpeg.py::test_whisperx_load_audio_uses_bundled_ffmpeg -v`
Expected: PASS (this validates the shim against the real whisperx decode). If it FAILS with an ffmpeg-not-found error, the shim in Task 2 is wrong — fix before continuing.

- [ ] **Step 3: Add the call in `transcribe()`**

In `tathurell/whisperx_core.py`, add the import near the top (with the other `tathurell` import):
```python
from tathurell.ffmpeg import ensure_ffmpeg_on_path
```
Then in `WhisperXTranscriber.transcribe`, make `ensure_ffmpeg_on_path()` the first line of the method body, immediately before `audio = whisperx.load_audio(audio_path)`:
```python
    def transcribe(self, audio_path: str) -> list:
        """Return [{"word", "start", "end", "speaker"}] for the audio file."""
        ensure_ffmpeg_on_path()  # bundled ffmpeg shadows any system one for load_audio
        audio = whisperx.load_audio(audio_path)
```

- [ ] **Step 4: Run the ffmpeg test module to confirm nothing regressed**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_ffmpeg.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tathurell/whisperx_core.py tests/test_ffmpeg.py
git commit -m "tathurell: activate bundled ffmpeg before whisperx.load_audio"
```

---

### Task 5: Move pydub runtime→eval; final verification

**Files:**
- Modify: `requirements.txt` (remove pydub)
- Modify: `eval/requirements-eval.txt` (add pydub)

- [ ] **Step 1: Confirm no runtime code imports pydub anymore**

Run: `grep -rn "pydub" tathurell/ tathurell_transcribe.py`
Expected: **no output** (the only runtime user, `extract_clip`, was rewritten in Task 3). If anything prints, stop and fix it before proceeding.

- [ ] **Step 2: Remove pydub from `requirements.txt`**

Delete the `pydub==0.25.1` line from `requirements.txt`.

- [ ] **Step 3: Add pydub to `eval/requirements-eval.txt`**

In `eval/requirements-eval.txt`, add `pydub` under the "Metrics / corpora / test harness" group (the eval-only vosk engine `eval/engines/vosk_pyannote.py` imports it). For example, add it after the `soundfile` line:
```
soundfile
pydub
```

- [ ] **Step 4: Run the full suite**

Run: `env -u HF_TOKEN $PY -m pytest`
Expected: all previously-passing tests still pass, plus the 4 new tests in `test_ffmpeg.py` and the 1 new test in `test_sampling_clip.py` — i.e. `54 passed, 5 skipped` (was `49 passed, 5 skipped`; +5 added). Confirm 0 failures.

- [ ] **Step 5: Final clean-machine sanity check**

Run (proves both consumers work with NO system ffmpeg on PATH):
```bash
env -u HF_TOKEN PATH="/nonexistent-tathurell-dir" $PY - <<'PYEOF'
import tempfile, os, wave
from tathurell.sampling import extract_clip
from tathurell.ffmpeg import ensure_ffmpeg_on_path
import whisperx
out = os.path.join(tempfile.mkdtemp(), "c.wav")
extract_clip("dollop_test_a.mp3", 1.0, 4.0, out)
with wave.open(out) as w:
    print("clip dur:", round(w.getnframes()/w.getframerate(), 2), "s")
ensure_ffmpeg_on_path()
print("load_audio samples:", len(whisperx.load_audio("dollop_test_a.mp3")))
print("OK: bundled ffmpeg serves both consumers with no system ffmpeg")
PYEOF
```
Expected: `clip dur: ~3.0 s`, a large `load_audio samples` count, and the final `OK` line — no ffmpeg-not-found errors. (Note: `PATH` here has no shell utilities, so the script uses only Python + the bundled binary by absolute path — that's the point.)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt eval/requirements-eval.txt
git commit -m "packaging: pydub is now eval-only (runtime decodes via bundled ffmpeg)"
```

---

## Notes for the implementer

- **Why the symlink shim (Task 2):** whisperx calls `subprocess.run(["ffmpeg", ...])`; the bundled binary is named `ffmpeg-<plat>-<ver>`, so only a file literally named `ffmpeg` on PATH satisfies it. `extract_clip` sidesteps this entirely by calling the bundled binary's absolute path directly.
- **Why drop pydub (Task 3):** `imageio-ffmpeg` ships ffmpeg but not ffprobe; pydub's `from_file` autodetects format via ffprobe. A direct ffmpeg call we control avoids that gap. pydub stays for the eval-only vosk engine.
- **No decode-fidelity risk:** `load_audio` is untouched (same ffmpeg, same flags), so transcription output is identical; the change is purely *which* ffmpeg binary runs.
