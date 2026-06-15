# Speaker-Naming Modal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the terminal `input()` speaker-naming loop with a local browser modal that plays an ~8-second audio sample of each detected speaker beside a name field.

**Architecture:** Two new units in `tathurell/` — `sampling.py` (pure longest-run sample picker + a pydub clip extractor) and `naming_ui.py` (a Flask one-page modal served on a free localhost port, block-until-submit) — wired into `tathurell_transcribe.py` where `prompt_names` is today, with `prompt_names` kept as the `--no-ui`/headless fallback.

**Tech Stack:** Python 3.10 in the `tathurell-eval` venv, Flask + werkzeug, pydub (already present), soundfile, pytest. All commands use `~/.pyenv/versions/tathurell-eval/bin/python`.

Spec: `docs/superpowers/specs/2026-06-15-speaker-naming-modal-design.md`.

---

## File structure

```
tathurell/
  sampling.py     # pick_speaker_samples (PURE) + extract_clip (pydub I/O)
  naming_ui.py    # create_app (Flask routes) + collect_names (server lifecycle)
tathurell_transcribe.py   # add --no-ui + resolve_names() (UI path | prompt_names fallback)
eval/requirements-eval.txt  # add flask
tests/
  test_sampling.py        # PURE unit tests for pick_speaker_samples
  test_sampling_clip.py   # clip extraction smoke
  test_naming_ui.py       # Flask test-client: GET / render, POST /submit
  test_cli_resolve_names.py  # --no-ui + fallback routing (PURE, monkeypatched)
```

---

## Task 1: Add Flask dependency

**Files:**
- Modify: `eval/requirements-eval.txt`

- [ ] **Step 1: Install Flask into the eval venv**

```bash
~/.pyenv/versions/tathurell-eval/bin/pip install flask
```
Expected: installs Flask + werkzeug (werkzeug is a Flask dependency).

- [ ] **Step 2: Confirm import**

```bash
~/.pyenv/versions/tathurell-eval/bin/python -c "import flask, werkzeug; print('flask', flask.__version__)"
```
Expected: prints a flask version.

- [ ] **Step 3: Record it in requirements**

Append a line `flask` to `eval/requirements-eval.txt` (keep the existing lines).

- [ ] **Step 4: Commit**

```bash
git add eval/requirements-eval.txt
git commit -m "deps: add flask for the speaker-naming modal"
```

---

## Task 2: `pick_speaker_samples` (pure)

**Files:**
- Create: `tathurell/sampling.py`
- Test: `tests/test_sampling.py`

- [ ] **Step 1: Write the failing test** (`tests/test_sampling.py`)

```python
from tathurell.sampling import pick_speaker_samples


def test_picks_longest_run_not_first():
    # A speaks a short run, then B, then A again in a long run.
    words = [
        {"word": "hi", "start": 0.0, "end": 0.5, "speaker": "A"},
        {"word": "yo", "start": 0.5, "end": 1.0, "speaker": "B"},
        {"word": "this", "start": 1.0, "end": 1.4, "speaker": "A"},
        {"word": "is", "start": 1.4, "end": 1.8, "speaker": "A"},
        {"word": "longer", "start": 1.8, "end": 3.0, "speaker": "A"},
    ]
    out = pick_speaker_samples(words)
    # A's longest run is the 3-word one starting at 1.0, not the 1-word "hi" at 0.0
    assert out["A"]["start"] == 1.0
    assert out["A"]["text"] == "this is longer"
    assert out["B"]["text"] == "yo"


def test_caps_to_max_seconds():
    words = [
        {"word": "a", "start": 0.0, "end": 1.0, "speaker": "A"},
        {"word": "b", "start": 1.0, "end": 2.0, "speaker": "A"},
        {"word": "c", "start": 2.0, "end": 30.0, "speaker": "A"},
    ]
    out = pick_speaker_samples(words, max_seconds=8.0)
    assert out["A"]["start"] == 0.0
    assert out["A"]["end"] == 8.0  # capped at start + 8
    assert out["A"]["text"] == "a b"  # "c" starts at 2.0 (<8) but... see note


def test_ignores_none_speaker_words():
    words = [
        {"word": "x", "start": 0.0, "end": 0.2, "speaker": None},
        {"word": "y", "start": 0.2, "end": 1.0, "speaker": "A"},
    ]
    out = pick_speaker_samples(words)
    assert set(out) == {"A"}
    assert out["A"]["text"] == "y"
```

NOTE for Step 3: in `test_caps_to_max_seconds`, "c" starts at 2.0 which is `< 8.0`, so by the
"words whose start is within the capped window" rule it WOULD be included → text "a b c". Adjust the
assertion to `"a b c"` if that's the behavior you implement; the load-bearing assertions are
`start==0.0` and `end==8.0` (the cap). Pick one and make the test match the implementation in Step 3.

- [ ] **Step 2: Run test, verify it fails**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_sampling.py -v`
Expected: FAIL (ImportError: cannot import name 'pick_speaker_samples').

- [ ] **Step 3: Implement `pick_speaker_samples` in `tathurell/sampling.py`**

```python
"""Pick a representative audio sample per speaker, and extract the clip.

pick_speaker_samples is pure (operates on the word list). extract_clip does I/O.
"""


def pick_speaker_samples(words, max_seconds=8.0):
    """For each speaker, return their longest contiguous run as a sample window.

    Returns {speaker: {"start": float, "end": float, "text": str}}. The window is
    the longest run's first-word start to either the run end or start+max_seconds,
    whichever is sooner; text is the run's words whose start falls in that window.
    Words with speaker None are ignored (not a nameable speaker).
    """
    # Build consecutive same-speaker runs (skipping None-speaker words).
    runs = []  # list of (speaker, [word, ...])
    for w in words:
        spk = w.get("speaker")
        if spk is None:
            continue
        if runs and runs[-1][0] == spk:
            runs[-1][1].append(w)
        else:
            runs.append((spk, [w]))

    # Longest run per speaker, by wall-clock duration.
    best = {}  # speaker -> (duration, run_words)
    for spk, ws in runs:
        dur = ws[-1]["end"] - ws[0]["start"]
        if spk not in best or dur > best[spk][0]:
            best[spk] = (dur, ws)

    out = {}
    for spk, (_dur, ws) in best.items():
        start = ws[0]["start"]
        cap_end = start + max_seconds
        end = min(ws[-1]["end"], cap_end)
        text = " ".join(w["word"] for w in ws if w["start"] < cap_end)
        out[spk] = {"start": start, "end": end, "text": text}
    return out
```

(With this implementation, `test_caps_to_max_seconds` text is `"a b c"` — set that assertion.)

- [ ] **Step 4: Run test, verify it passes**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_sampling.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tathurell/sampling.py tests/test_sampling.py
git commit -m "tathurell: pick_speaker_samples (longest-run sample per speaker)"
```

---

## Task 3: `extract_clip` (pydub I/O) + smoke

**Files:**
- Modify: `tathurell/sampling.py`
- Test: `tests/test_sampling_clip.py`

- [ ] **Step 1: Add `extract_clip` to `tathurell/sampling.py`**

```python
def extract_clip(audio_path, start, end, out_path):
    """Write the [start, end] second slice of audio_path to out_path as a WAV.

    Uses pydub (ffmpeg) so any input format (mp3/wav/m4a...) works; pydub indexes
    in milliseconds.
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path)
    clip = audio[int(start * 1000):int(end * 1000)]
    clip.export(out_path, format="wav")
```

- [ ] **Step 2: Write the smoke test** (`tests/test_sampling_clip.py`)

```python
import os
import wave
from tathurell.sampling import extract_clip


def test_extract_clip_writes_wav_of_expected_duration(tmp_path):
    out = tmp_path / "clip.wav"
    extract_clip("dollop_test_a.mp3", 1.0, 4.0, str(out))
    assert out.exists() and out.stat().st_size > 0
    with wave.open(str(out)) as w:
        dur = w.getnframes() / w.getframerate()
    assert 2.7 < dur < 3.3  # ~3s window
```

- [ ] **Step 3: Run the smoke test**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_sampling_clip.py -v`
Expected: PASS (reads the mp3 via pydub/ffmpeg, writes a ~3s wav).

- [ ] **Step 4: Commit**

```bash
git add tathurell/sampling.py tests/test_sampling_clip.py
git commit -m "tathurell: extract_clip (format-agnostic audio slice via pydub)"
```

---

## Task 4: Flask routes (`create_app`) — test-client driven

**Files:**
- Create: `tathurell/naming_ui.py`
- Test: `tests/test_naming_ui.py`

- [ ] **Step 1: Write the failing test** (`tests/test_naming_ui.py`)

```python
import threading
from tathurell.naming_ui import create_app


def _samples():
    return {
        "SPEAKER_00": {"start": 0.0, "end": 8.0, "text": "hello there"},
        "SPEAKER_01": {"start": 9.0, "end": 12.0, "text": "good morning"},
    }


def test_index_renders_field_and_audio_per_speaker(tmp_path):
    app = create_app(_samples(), str(tmp_path), {}, threading.Event())
    html = app.test_client().get("/").get_data(as_text=True)
    for spk in ("SPEAKER_00", "SPEAKER_01"):
        assert f'name="{spk}"' in html        # a name input per speaker
        assert f"/clip/{spk}" in html          # an audio source per speaker
    assert "hello there" in html               # sample text shown


def test_submit_fills_result_and_sets_done(tmp_path):
    result, done = {}, threading.Event()
    app = create_app(_samples(), str(tmp_path), result, done)
    app.test_client().post("/submit", data={"SPEAKER_00": "Dave", "SPEAKER_01": "  "})
    assert result == {"SPEAKER_00": "Dave", "SPEAKER_01": "SPEAKER_01"}  # blank -> label
    assert done.is_set()
```

- [ ] **Step 2: Run test, verify it fails**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_naming_ui.py -v`
Expected: FAIL (ImportError: cannot import name 'create_app').

- [ ] **Step 3: Implement `create_app` in `tathurell/naming_ui.py`**

```python
"""Local browser modal for naming speakers by ear.

create_app builds the Flask app (pure-ish, test-client friendly). collect_names
(Task 5) extracts clips, runs the app on a free port, opens the browser, and
blocks until the form is submitted.
"""
import os

from flask import Flask, request, send_file

_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Name the speakers</title>
<style>body{{font-family:sans-serif;max-width:760px;margin:2rem auto}}
.row{{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}}
.txt{{color:#444;font-style:italic;margin:.5rem 0}} input{{font-size:1rem;padding:.3rem}}</style>
</head><body><h2>Who is each speaker?</h2><form method="post" action="/submit">
{rows}
<button type="submit" style="font-size:1rem;padding:.5rem 1rem">Save names</button>
</form></body></html>"""

_ROW = """<div class="row"><b>{spk}</b>
<audio controls src="/clip/{spk}"></audio>
<div class="txt">"{text}"</div>
<label>Name: <input name="{spk}" placeholder="{spk}"></label></div>"""


def create_app(samples, clip_dir, result, done):
    """Flask app for the naming modal.

    samples: {speaker: {"start","end","text"}}; clip_dir: holds <speaker>.wav;
    result: dict filled with {speaker: name} on submit; done: Event set on submit.
    """
    app = Flask(__name__)

    @app.route("/")
    def index():
        rows = "".join(
            _ROW.format(spk=spk, text=s["text"].replace('"', "&quot;"))
            for spk, s in samples.items()
        )
        return _PAGE.format(rows=rows)

    @app.route("/clip/<speaker>")
    def clip(speaker):
        # speaker comes from our own keys; guard against path traversal anyway.
        if speaker not in samples:
            return ("unknown speaker", 404)
        return send_file(os.path.join(clip_dir, f"{speaker}.wav"), mimetype="audio/wav")

    @app.route("/submit", methods=["POST"])
    def submit():
        for spk in samples:
            name = request.form.get(spk, "").strip()
            result[spk] = name or spk  # blank -> fall back to the label
        done.set()
        return "<p>Names saved. You can close this tab.</p>"

    return app
```

- [ ] **Step 4: Run test, verify it passes**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_naming_ui.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tathurell/naming_ui.py tests/test_naming_ui.py
git commit -m "tathurell: naming modal Flask routes (index, clip, submit)"
```

---

## Task 5: `collect_names` (server lifecycle)

**Files:**
- Modify: `tathurell/naming_ui.py`

- [ ] **Step 1: Add `collect_names` to `tathurell/naming_ui.py`**

```python
import shutil
import sys
import tempfile
import threading
import webbrowser

from werkzeug.serving import make_server

from tathurell.sampling import extract_clip


def collect_names(samples, audio_path, open_browser=True):
    """Extract a clip per speaker, serve the modal, block until submit; return names.

    Returns {speaker: name}. On Ctrl-C (tab closed without submitting) falls back
    to using each speaker's label as its name.
    """
    clip_dir = tempfile.mkdtemp(prefix="tathurell_clips_")
    result, done = {}, threading.Event()
    try:
        for spk, s in samples.items():
            extract_clip(audio_path, s["start"], s["end"], os.path.join(clip_dir, f"{spk}.wav"))
        app = create_app(samples, clip_dir, result, done)
        server = make_server("127.0.0.1", 0, app)  # port 0 -> OS picks a free port
        url = f"http://127.0.0.1:{server.server_port}/"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[tathurell] Name the speakers at {url} (opening browser)...", file=sys.stderr)
        if open_browser:
            webbrowser.open(url)
        try:
            done.wait()
        except KeyboardInterrupt:
            result = {spk: spk for spk in samples}
        server.shutdown()
        return result
    finally:
        shutil.rmtree(clip_dir, ignore_errors=True)
```

- [ ] **Step 2: Verify the module imports**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -c "from tathurell.naming_ui import create_app, collect_names; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Manual browser check** (the one click-through that can't be unit-tested)

```bash
~/.pyenv/versions/tathurell-eval/bin/python - <<'PY'
from tathurell.naming_ui import collect_names
samples = {"SPEAKER_00": {"start": 0.0, "end": 8.0, "text": "the spanish version of the dollop"}}
# extract_clip will read dollop_test_a.mp3 for the clip
print(collect_names(samples, "dollop_test_a.mp3"))
PY
```
Expected: a browser tab opens with a player + name field; after you type a name and Save, the script prints `{"SPEAKER_00": "<your name>"}`. (If running headless, skip — covered by the fallback in Task 6.)

- [ ] **Step 4: Commit**

```bash
git add tathurell/naming_ui.py
git commit -m "tathurell: collect_names — serve modal on free port, block until submit"
```

---

## Task 6: Wire into the CLI + `--no-ui` fallback

**Files:**
- Modify: `tathurell_transcribe.py`
- Test: `tests/test_cli_resolve_names.py`

- [ ] **Step 1: Write the failing test** (`tests/test_cli_resolve_names.py`)

```python
import importlib


def test_no_ui_uses_prompt_names(monkeypatch):
    cli = importlib.import_module("tathurell_transcribe")
    monkeypatch.setattr(cli, "prompt_names", lambda groups: {"_": "from_prompt"})
    out = cli.resolve_names(words=[], groups=[{"speaker": "A", "text": "x"}],
                            audio_path="a.wav", no_ui=True)
    assert out == {"_": "from_prompt"}


def test_ui_failure_falls_back_to_prompt(monkeypatch):
    cli = importlib.import_module("tathurell_transcribe")
    monkeypatch.setattr(cli, "prompt_names", lambda groups: {"_": "fallback"})

    def boom(*a, **k):
        raise RuntimeError("no display")

    # collect_names is imported inside resolve_names; patch it at its source.
    import tathurell.naming_ui as ui
    monkeypatch.setattr(ui, "collect_names", boom)
    out = cli.resolve_names(words=[{"word": "x", "start": 0.0, "end": 1.0, "speaker": "A"}],
                            groups=[{"speaker": "A", "text": "x"}],
                            audio_path="a.wav", no_ui=False)
    assert out == {"_": "fallback"}
```

- [ ] **Step 2: Run test, verify it fails**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_cli_resolve_names.py -v`
Expected: FAIL (AttributeError: module has no attribute 'resolve_names').

- [ ] **Step 3: Edit `tathurell_transcribe.py`** — add `resolve_names`, the `--no-ui` arg, and call it

Add this function (above `main`):

```python
def resolve_names(words, groups, audio_path, no_ui):
    """Get {speaker: name}: browser modal by default; terminal prompts on --no-ui
    or if the modal can't run (headless/no browser)."""
    if no_ui:
        return prompt_names(groups)
    try:
        from tathurell.sampling import pick_speaker_samples
        from tathurell.naming_ui import collect_names
        return collect_names(pick_speaker_samples(words), audio_path)
    except Exception as exc:
        print(f"[tathurell] naming modal unavailable ({exc}); using terminal prompts",
              file=sys.stderr)
        return prompt_names(groups)
```

In `main`, add the arg (next to `--output`/`--model`):

```python
    ap.add_argument("--no-ui", action="store_true",
                    help="skip the browser naming modal; name speakers via terminal prompts")
```

And replace the naming line `names = prompt_names(groups)` with:

```python
    names = resolve_names(words, groups, args.audio_path, args.no_ui)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `~/.pyenv/versions/tathurell-eval/bin/python -m pytest tests/test_cli_resolve_names.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify argparse still works and `--no-ui` is present**

Run: `~/.pyenv/versions/tathurell-eval/bin/python tathurell_transcribe.py --help`
Expected: usage shows `audio_path`, `--output`, `--model`, `--no-ui`.

- [ ] **Step 6: Headless end-to-end with the fallback (no browser)** — token-gated

Run:
```bash
HF_TOKEN=$(tr -d '[:space:]' < eval/data/.hf_token) bash -c \
  'yes "spk" | ~/.pyenv/versions/tathurell-eval/bin/python tathurell_transcribe.py dollop_test_a.mp3 --no-ui --output /tmp/tath_noui.txt'
test -s /tmp/tath_noui.txt && head -2 /tmp/tath_noui.txt
```
Expected: `--no-ui` path runs the terminal prompts (auto-answered "spk"), writes a non-empty file with `spk: ...` lines. Confirms the fallback path is intact.

- [ ] **Step 7: Full repo suite (no token; gated/browser steps skip)**

Run: `env -u HF_TOKEN ~/.pyenv/versions/tathurell-eval/bin/python -m pytest -q`
Expected: all pure tests pass (sampling, naming_ui, cli_resolve_names, plus existing); engine smokes skip.

- [ ] **Step 8: Commit**

```bash
git add tathurell_transcribe.py tests/test_cli_resolve_names.py
git commit -m "tathurell: wire naming modal into CLI with --no-ui fallback"
```

---

## Self-Review

**Spec coverage:**
- §2 `sampling.py` `pick_speaker_samples` (longest run, 8s cap, ignore None) → Task 2. ✓
- §2 `sampling.py` `extract_clip` (format-agnostic, WAV out) → Task 3. ✓
- §2 `naming_ui.py` routes (`GET /`, `GET /clip/<speaker>`, `POST /submit`; blank→label) → Task 4. ✓
- §2 `naming_ui.py` `collect_names` (free port, background thread, open browser, block-until-submit, cleanup) → Task 5. ✓
- §2/§5 CLI wiring + `--no-ui` + auto-fallback on failure → Task 6. ✓
- §4 Flask dep → Task 1. ✓
- §5 fallback (`--no-ui`, exception → prompt_names; blank name → label; Ctrl-C → labels; free port via bind 0) → Tasks 4/5/6. ✓
- §6 testing (pure sample tests, Flask test-client submit/render, clip smoke, CLI routing; manual browser) → Tasks 2/3/4/6 + Task 5 Step 3. ✓
- §1/§7 output format unchanged, persistence out of scope → no task touches `apply_names`/output format or adds persistence. ✓

**Placeholder scan:** No "TBD"/"implement later". The one judgment call (`test_caps_to_max_seconds` text = "a b c" vs "a b") is resolved explicitly in Task 2 Step 3's note — implement the inclusive rule, assert `"a b c"`.

**Type consistency:** `pick_speaker_samples(words, max_seconds) -> {spk: {"start","end","text"}}` (Task 2) is consumed by `collect_names(samples, audio_path)` (Task 5) and `create_app(samples, clip_dir, result, done)` (Task 4) — same `samples` shape. `extract_clip(audio_path, start, end, out_path)` (Task 3) called by `collect_names` with `s["start"]/s["end"]` (Task 5) — matches. `resolve_names(words, groups, audio_path, no_ui)` (Task 6) calls `prompt_names(groups)` (existing) and `collect_names(pick_speaker_samples(words), audio_path)` — signatures match. `result`/`done` (dict + threading.Event) consistent between `create_app` and `collect_names`.
