# Front-Door Transcription UI (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A persistent local web app that drives the whole pipeline from the browser: upload audio → transcribe with coarse stage progress → name speakers (reusing today's naming) → preview + download.

**Architecture:** New `tathurell/webapp.py` = a persistent Flask app + a single in-process lock-guarded `Job` state + a background worker thread + a single-page HTML/JS shell + a `python -m` launcher. Transcription runs off the request thread; the SPA polls JSON endpoints. The job runner takes an **injectable transcriber factory** (default `WhisperXTranscriber`) so the full flow is testable with a fake — no model, no HF token.

**Tech Stack:** Python 3.10 in the `tathurell-eval` venv, Flask + werkzeug, vanilla JS (no framework), pytest. Reuses `pick_speaker_samples`/`extract_clip` (`sampling.py`), `group_by_speaker`/`apply_names` (`naming.py`), `WhisperXTranscriber` (`whisperx_core.py`). Spec: `docs/superpowers/specs/2026-06-15-frontdoor-ui-phase1-design.md`.

**File structure:**
- Modify `tathurell/whisperx_core.py` — add optional `progress` callback to `transcribe`.
- Create `tathurell/webapp.py` — the Phase 1 app (Job state, routes, job runner, SPA shell, launcher).
- Create `tests/test_webapp.py` — flow tests with a fake transcriber.
- Modify `tests/test_whisperx_core_smoke.py` — extend the gated smoke to assert progress stages.

> Throughout: `PY=~/.pyenv/versions/tathurell-eval/bin/python`. Run pytest as `env -u HF_TOKEN $PY -m pytest` from the repo root (where `dollop_test_a.mp3` lives). Commit messages end with the trailer:
> `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: Add `progress` callback to `WhisperXTranscriber.transcribe`

**Files:**
- Modify: `tathurell/whisperx_core.py` (`transcribe`, ~lines 57-82)
- Test: `tests/test_whisperx_core_smoke.py`

- [ ] **Step 1: Write the interface test + extend the gated smoke** — replace the contents of `tests/test_whisperx_core_smoke.py` with:

```python
import inspect
import os

import pytest

from tathurell.whisperx_core import WhisperXTranscriber


def test_transcribe_accepts_progress_param():
    # Lock the interface without needing models: progress is an optional kwarg.
    sig = inspect.signature(WhisperXTranscriber.transcribe)
    assert "progress" in sig.parameters
    assert sig.parameters["progress"].default is None


@pytest.mark.skipif("HF_TOKEN" not in os.environ, reason="needs HF_TOKEN (pyannote)")
def test_transcriber_runs_on_clip_and_reports_progress():
    stages = []
    words = WhisperXTranscriber().transcribe("dollop_test_a.mp3", progress=stages.append)
    assert len(words) > 50
    assert all({"word", "start", "end", "speaker"} <= set(w) for w in words)
    # Coarse stages fire in order (each at most once, monotonic through the pipeline).
    assert stages == ["transcribing", "aligning", "diarizing", "finishing"]
```

- [ ] **Step 2: Run the interface test (runs without a token) to verify it fails**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_whisperx_core_smoke.py::test_transcribe_accepts_progress_param -v`
Expected: FAIL — `transcribe` has no `progress` parameter yet.

- [ ] **Step 3: Add the callback in `transcribe`** — in `tathurell/whisperx_core.py`, change the method signature and insert four `progress(...)` calls at the existing stage boundaries. The method becomes:

```python
    def transcribe(self, audio_path: str, progress=None) -> list:
        """Return [{"word", "start", "end", "speaker"}] for the audio file.

        progress: optional callback(stage_name) invoked at each coarse pipeline
        stage ("transcribing"/"aligning"/"diarizing"/"finishing"). Default None
        (the CLI passes nothing -> unchanged behavior).
        """
        def _p(stage):
            if progress is not None:
                progress(stage)

        ensure_ffmpeg_on_path()  # bundled ffmpeg shadows any system one for load_audio
        audio = whisperx.load_audio(audio_path)
        _p("transcribing")
        result = self._model.transcribe(audio, batch_size=8)
        _p("aligning")
        align_model, meta = whisperx.load_align_model(
            language_code=result["language"], device=self._device
        )
        result = whisperx.align(result["segments"], align_model, meta, audio, self._device)
        _p("diarizing")
        diar = self._diarize(audio)
        _p("finishing")
        # fill_nearest=True so words in a diarization gap get the nearest speaker
        # instead of None (whisperx default leaves them unassigned).
        result = whisperx.assign_word_speakers(diar, result, fill_nearest=True)
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
        # whisperx assigns each word independently, so a single word at a turn
        # boundary can flip speaker mid-sentence. Realign per sentence by majority.
        return realign_speakers(words)
```

(Preserve the existing `ensure_ffmpeg_on_path()` call and all downstream logic exactly — only the signature, docstring, the `_p` helper, and the four `_p(...)` calls are added.)

- [ ] **Step 4: Run the interface test to verify it passes**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_whisperx_core_smoke.py -v`
Expected: PASS for `test_transcribe_accepts_progress_param`; `test_transcriber_runs_on_clip_and_reports_progress` SKIPPED (no token).

- [ ] **Step 5: Commit**

```bash
git add tathurell/whisperx_core.py tests/test_whisperx_core_smoke.py
git commit -m "tathurell: optional progress callback on transcribe (coarse stages)"
```

---

### Task 2: `Job` state model in `tathurell/webapp.py`

**Files:**
- Create: `tathurell/webapp.py`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_webapp.py`:

```python
import os

from tathurell.webapp import Job


def test_job_starts_idle():
    job = Job()
    assert job.snapshot() == {"stage": "idle"}


def test_job_lifecycle_to_naming_and_done():
    job = Job()
    job.start("/tmp/does-not-matter", "meeting.mp3")
    assert job.snapshot()["stage"] == "transcribing"
    job.set_stage("diarizing")
    assert job.snapshot()["stage"] == "diarizing"
    groups = [{"speaker": "SPEAKER_00", "text": "hello there"}]
    samples = {"SPEAKER_00": {"start": 0.0, "end": 1.0, "text": "hello there"}}
    job.set_naming(groups, samples)
    snap = job.snapshot()
    assert snap["stage"] == "naming"
    assert snap["speakers"] == [{"id": "SPEAKER_00", "text": "hello there"}]
    job.set_done("Alice: hello there")
    assert job.snapshot()["stage"] == "done"
    assert job.text == "Alice: hello there"


def test_job_error():
    job = Job()
    job.start("/tmp/x", "a.wav")
    job.set_error("boom")
    snap = job.snapshot()
    assert snap["stage"] == "error"
    assert snap["error"] == "boom"


def test_reset_clears_state_and_tmpdir(tmp_path):
    job = Job()
    d = tmp_path / "clips"
    d.mkdir()
    (d / "x.wav").write_bytes(b"x")
    job.start(str(d), "a.wav")
    job.reset()
    assert job.snapshot() == {"stage": "idle"}
    assert not d.exists()  # tmpdir removed
```

- [ ] **Step 2: Run to verify failure**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tathurell.webapp'`.

- [ ] **Step 3: Create `tathurell/webapp.py` with the `Job` class** (module header + imports + Job; routes/runner/shell come in later tasks):

```python
"""Front-door transcription web app (Phase 1).

upload -> transcribe (background job, coarse stage progress) -> name speakers
-> inline preview + download. Persistent, one-shot, single job at a time.
Launched via `python -m tathurell.webapp`. The CLI (tathurell_transcribe.py) and
naming_ui.collect_names are unaffected.
"""
import os
import shutil
import sys
import tempfile
import threading
import webbrowser

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.serving import make_server

from tathurell.naming import apply_names, group_by_speaker
from tathurell.sampling import extract_clip, pick_speaker_samples
from tathurell.whisperx_core import WhisperXTranscriber

# Stages reported while the background job runs (before naming).
_RUNNING_STAGES = ("transcribing", "aligning", "diarizing", "finishing")


class Job:
    """Single in-process transcription job: one-shot, lock-guarded state.

    Lifecycle of `stage`: idle -> transcribing/aligning/diarizing/finishing
    -> naming -> done (or -> error at any point). reset() returns to idle and
    removes the job's temp dir.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self._tmpdir = None
        self._init_state()

    def _init_state(self):
        self.stage = "idle"
        self.error = None
        self.audio_name = None
        self.samples = None  # {speaker: {"start","end","text"}}
        self.groups = None   # [{"speaker","text"}]
        self.text = None     # final named transcript

    @property
    def tmpdir(self):
        return self._tmpdir

    def start(self, tmpdir, audio_name):
        with self.lock:
            self._tmpdir = tmpdir
            self.audio_name = audio_name
            self.error = None
            self.stage = "transcribing"

    def set_stage(self, stage):
        with self.lock:
            self.stage = stage

    def set_naming(self, groups, samples):
        with self.lock:
            self.groups = groups
            self.samples = samples
            self.stage = "naming"

    def set_done(self, text):
        with self.lock:
            self.text = text
            self.stage = "done"

    def set_error(self, message):
        with self.lock:
            self.stage = "error"
            self.error = message

    def reset(self):
        with self.lock:
            if self._tmpdir:
                shutil.rmtree(self._tmpdir, ignore_errors=True)
                self._tmpdir = None
            self._init_state()

    def snapshot(self):
        """JSON-safe view for GET /status."""
        with self.lock:
            snap = {"stage": self.stage}
            if self.error:
                snap["error"] = self.error
            if self.stage == "naming" and self.samples:
                snap["speakers"] = [
                    {"id": spk, "text": s["text"]} for spk, s in self.samples.items()
                ]
            return snap
```

- [ ] **Step 4: Run to verify pass**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tathurell/webapp.py tests/test_webapp.py
git commit -m "tathurell: webapp Job state model (one-shot, lock-guarded)"
```

---

### Task 3: Flask app skeleton — `create_app`, `GET /`, `GET /status`, `POST /reset`

**Files:**
- Modify: `tathurell/webapp.py`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_webapp.py`:

```python
from tathurell.webapp import create_app


def test_index_serves_shell():
    app = create_app()
    r = app.test_client().get("/")
    assert r.status_code == 200
    assert b"<!doctype html>" in r.data.lower()


def test_status_starts_idle_and_reset_works():
    app = create_app()
    c = app.test_client()
    assert c.get("/status").get_json() == {"stage": "idle"}
    assert c.post("/reset").status_code == 200
    assert c.get("/status").get_json() == {"stage": "idle"}
```

- [ ] **Step 2: Run to verify failure**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -k "index or status_starts" -v`
Expected: FAIL — `cannot import name 'create_app'`.

- [ ] **Step 3: Add `create_app` + a placeholder shell** — append to `tathurell/webapp.py`:

```python
# Replaced with the full single-page app in a later task; kept as a constant so
# the route never changes.
_SHELL = "<!doctype html><html><body>tathurell</body></html>"


def create_app(transcriber_factory=WhisperXTranscriber):
    """Build the front-door app. transcriber_factory is injected so tests can
    supply a fake (no model / no HF token)."""
    app = Flask(__name__)
    job = Job()
    app.config["JOB"] = job  # exposed for tests

    @app.route("/")
    def index():
        return _SHELL

    @app.route("/status")
    def status():
        return jsonify(job.snapshot())

    @app.route("/reset", methods=["POST"])
    def reset():
        job.reset()
        return ("", 200)

    return app
```

- [ ] **Step 4: Run to verify pass**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add tathurell/webapp.py tests/test_webapp.py
git commit -m "tathurell: webapp skeleton (create_app, /, /status, /reset)"
```

---

### Task 4: `POST /upload` + background job runner

**Files:**
- Modify: `tathurell/webapp.py`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing tests** — add the fake transcriber + a poll helper + tests at the TOP of `tests/test_webapp.py` (place `FakeTranscriber`/`_poll` near the imports so later tasks reuse them):

```python
import time


class FakeTranscriber:
    """Stand-in for WhisperXTranscriber: fires the stage callbacks and returns
    canned words whose timestamps fall inside dollop_test_a.mp3 (so extract_clip
    produces real clips). No model, no token."""

    def transcribe(self, audio_path, progress=None):
        for stage in ("transcribing", "aligning", "diarizing", "finishing"):
            if progress is not None:
                progress(stage)
        return [
            {"word": "the", "start": 0.2, "end": 0.4, "speaker": "SPEAKER_00"},
            {"word": "spanish", "start": 0.4, "end": 0.9, "speaker": "SPEAKER_00"},
            {"word": "version", "start": 0.9, "end": 1.4, "speaker": "SPEAKER_00"},
            {"word": "welcome", "start": 2.0, "end": 2.4, "speaker": "SPEAKER_01"},
            {"word": "gentlemen", "start": 2.4, "end": 3.0, "speaker": "SPEAKER_01"},
        ]


def _poll(client, target, tries=200, delay=0.05):
    """Poll GET /status until stage == target (or 'error'); return the snapshot."""
    for _ in range(tries):
        snap = client.get("/status").get_json()
        if snap["stage"] in (target, "error"):
            return snap
        time.sleep(delay)
    raise AssertionError(f"stage never reached {target!r}; last={snap}")


def _upload(client, path="dollop_test_a.mp3"):
    with open(path, "rb") as f:
        return client.post(
            "/upload",
            data={"audio": (f, path)},
            content_type="multipart/form-data",
        )


def test_upload_runs_job_to_naming():
    app = create_app(transcriber_factory=FakeTranscriber)
    c = app.test_client()
    assert _upload(c).status_code == 202
    snap = _poll(c, "naming")
    assert snap["stage"] == "naming"
    assert {s["id"] for s in snap["speakers"]} == {"SPEAKER_00", "SPEAKER_01"}


def test_upload_rejected_when_busy():
    app = create_app(transcriber_factory=FakeTranscriber)
    c = app.test_client()
    _upload(c)
    # Immediately try again; whatever the stage, a second upload is refused until idle.
    r2 = _upload(c)
    assert r2.status_code == 409
    _poll(c, "naming")  # let the first job settle


def test_upload_without_file_is_rejected():
    app = create_app(transcriber_factory=FakeTranscriber)
    r = app.test_client().post("/upload", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
```

- [ ] **Step 2: Run to verify failure**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -k upload -v`
Expected: FAIL — there is no `/upload` route yet (404).

- [ ] **Step 3: Add the job runner + `/upload`** — in `tathurell/webapp.py`, add the module-level `_run_job` (above `create_app`) and the `/upload` route (inside `create_app`):

```python
def _run_job(job, transcriber_factory, audio_path):
    """Background worker: transcribe -> group + sample -> extract clips -> naming.
    Any exception is captured into the job as an error (never kills the server)."""
    try:
        words = transcriber_factory().transcribe(audio_path, progress=job.set_stage)
        groups = group_by_speaker(words)
        samples = pick_speaker_samples(words)
        for spk, s in samples.items():
            extract_clip(audio_path, s["start"], s["end"],
                         os.path.join(job.tmpdir, f"{spk}.wav"))
        job.set_naming(groups, samples)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        job.set_error(str(exc))
```

Inside `create_app`, add this route (alongside the others):

```python
    @app.route("/upload", methods=["POST"])
    def upload():
        if job.snapshot()["stage"] != "idle":
            return ("a job is already running", 409)
        f = request.files.get("audio")
        if not f or not f.filename:
            return ("no audio file", 400)
        tmpdir = tempfile.mkdtemp(prefix="tathurell_web_")
        audio_path = os.path.join(tmpdir, "input" + os.path.splitext(f.filename)[1])
        f.save(audio_path)
        job.start(tmpdir, f.filename)
        threading.Thread(
            target=_run_job, args=(job, transcriber_factory, audio_path), daemon=True
        ).start()
        return ("", 202)
```

- [ ] **Step 4: Run to verify pass**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -v`
Expected: PASS (9 passed). (The job thread does a real `extract_clip` via the bundled ffmpeg — a few hundred ms; `_poll` waits for it.)

- [ ] **Step 5: Commit**

```bash
git add tathurell/webapp.py tests/test_webapp.py
git commit -m "tathurell: webapp upload + background transcription job"
```

---

### Task 5: `GET /clip/<speaker>`, `POST /names`, `GET /result`, `GET /download`

**Files:**
- Modify: `tathurell/webapp.py`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_webapp.py`:

```python
import wave


def test_clip_serves_wav_and_unknown_404():
    app = create_app(transcriber_factory=FakeTranscriber)
    c = app.test_client()
    _upload(c)
    _poll(c, "naming")
    r = c.get("/clip/SPEAKER_00")
    assert r.status_code == 200
    assert r.mimetype == "audio/wav"
    assert c.get("/clip/NOPE").status_code == 404


def test_names_then_result_and_download():
    app = create_app(transcriber_factory=FakeTranscriber)
    c = app.test_client()
    _upload(c, )
    _poll(c, "naming")
    # Name one speaker; leave the other blank -> falls back to its label.
    assert c.post("/names", json={"SPEAKER_00": "Alice", "SPEAKER_01": ""}).status_code == 200
    assert c.get("/status").get_json()["stage"] == "done"

    res = c.get("/result").get_json()
    assert res["filename"] == "dollop_test_a.transcription.txt"
    assert "Alice: the spanish version" in res["text"]
    assert "SPEAKER_01: welcome gentlemen" in res["text"]

    dl = c.get("/download")
    assert dl.status_code == 200
    assert dl.mimetype == "text/plain"
    assert "dollop_test_a.transcription.txt" in dl.headers["Content-Disposition"]
    assert b"Alice: the spanish version" in dl.data
```

- [ ] **Step 2: Run to verify failure**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -k "clip or names_then" -v`
Expected: FAIL — those routes 404 (not defined yet).

- [ ] **Step 3: Add the routes + filename helper** — add `_out_name` at module level (near `_run_job`) and the four routes inside `create_app`:

```python
def _out_name(audio_name):
    """Output filename derived from the uploaded audio name."""
    stem = os.path.splitext(os.path.basename(audio_name or "transcript"))[0]
    return f"{stem}.transcription.txt"
```

Inside `create_app`:

```python
    @app.route("/clip/<speaker>")
    def clip(speaker):
        if not job.samples or speaker not in job.samples:
            return ("unknown speaker", 404)
        return send_file(os.path.join(job.tmpdir, f"{speaker}.wav"),
                         mimetype="audio/wav")

    @app.route("/names", methods=["POST"])
    def names():
        if job.groups is None or not job.samples:
            return ("no transcript to name", 409)
        data = request.get_json(silent=True) or {}
        resolved = {spk: ((data.get(spk) or "").strip() or spk) for spk in job.samples}
        job.set_done(apply_names(job.groups, resolved))
        return ("", 200)

    @app.route("/result")
    def result():
        if job.text is None:
            return ("no result yet", 409)
        return jsonify({"text": job.text, "filename": _out_name(job.audio_name)})

    @app.route("/download")
    def download():
        if job.text is None:
            return ("no result yet", 409)
        return Response(
            job.text,
            mimetype="text/plain",
            headers={"Content-Disposition":
                     f'attachment; filename="{_out_name(job.audio_name)}"'},
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -v`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add tathurell/webapp.py tests/test_webapp.py
git commit -m "tathurell: webapp clip/names/result/download routes"
```

---

### Task 6: Single-page shell (HTML + JS) + `python -m` launcher

**Files:**
- Modify: `tathurell/webapp.py`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing structural test** — append to `tests/test_webapp.py`:

```python
def test_shell_wires_all_views_and_endpoints():
    html = create_app().test_client().get("/").get_data(as_text=True)
    # The four views the JS swaps between:
    for view in ("view-upload", "view-working", "view-naming", "view-result"):
        assert f'id="{view}"' in html
    # The JS talks to every endpoint:
    for ep in ("/upload", "/status", "/names", "/result", "/download", "/reset", "/clip/"):
        assert ep in html
```

- [ ] **Step 2: Run to verify failure**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py::test_shell_wires_all_views_and_endpoints -v`
Expected: FAIL — the placeholder `_SHELL` has none of those ids/endpoints.

- [ ] **Step 3: Replace `_SHELL` with the full SPA and add the launcher** — in `tathurell/webapp.py`, replace the `_SHELL = "..."` placeholder line with the full page below, and add `main()` + the `__main__` guard at the end of the file:

```python
_SHELL = """<!doctype html><html><head><meta charset="utf-8">
<title>Tathurell — transcribe</title><style>
body{font-family:sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem}
.hidden{display:none}.row{border:1px solid #ccc;border-radius:8px;padding:1rem;margin:1rem 0}
.txt{color:#444;font-style:italic;margin:.5rem 0}input{font-size:1rem;padding:.3rem}
button{font-size:1rem;padding:.5rem 1rem}#err{color:#b00}
pre{white-space:pre-wrap;background:#f6f6f6;border-radius:8px;padding:1rem;max-height:50vh;overflow:auto}
</style></head><body><h2>Tathurell</h2>

<div id="view-upload">
  <p>Choose an audio file to transcribe.</p>
  <input type="file" id="file" accept="audio/*">
  <button id="go">Transcribe</button>
</div>

<div id="view-working" class="hidden">
  <p><span id="spin">◐</span> <span id="stage">Starting…</span></p>
  <p class="txt">This can take a few minutes for long audio.</p>
  <p id="err"></p>
  <button id="working-reset" class="hidden">Start over</button>
</div>

<div id="view-naming" class="hidden">
  <h3>Who is each speaker?</h3>
  <form id="names"></form>
  <button id="save">Save names</button>
</div>

<div id="view-result" class="hidden">
  <h3>Transcript</h3>
  <pre id="preview"></pre>
  <a id="download" href="/download"><button>⤓ Download transcript</button></a>
  <button id="restart">↺ Start over</button>
</div>

<script>
var LABELS={transcribing:"Transcribing…",aligning:"Aligning words…",
  diarizing:"Identifying speakers…",finishing:"Finishing…"};
var views=["upload","working","naming","result"];
function show(v){views.forEach(function(n){
  document.getElementById("view-"+n).classList.toggle("hidden",n!==v);});}
function el(id){return document.getElementById(id);}

el("go").onclick=function(){
  var f=el("file").files[0]; if(!f){alert("Pick a file first.");return;}
  var fd=new FormData(); fd.append("audio",f);
  fetch("/upload",{method:"POST",body:fd}).then(function(r){
    if(!r.ok){r.text().then(function(t){alert(t);});return;}
    el("err").textContent=""; el("working-reset").classList.add("hidden");
    show("working"); poll();});
};

function poll(){
  fetch("/status").then(function(r){return r.json();}).then(function(s){
    if(s.stage==="error"){el("stage").textContent="Something went wrong.";
      el("err").textContent=s.error||""; el("working-reset").classList.remove("hidden");return;}
    if(["transcribing","aligning","diarizing","finishing"].indexOf(s.stage)>=0){
      el("stage").textContent=LABELS[s.stage]; show("working"); setTimeout(poll,1000);return;}
    if(s.stage==="naming"){renderNaming(s.speakers); show("naming");return;}
    if(s.stage==="done"){loadResult();return;}
  });
}

function renderNaming(speakers){
  var form=el("names"); form.innerHTML="";
  speakers.forEach(function(sp){
    var div=document.createElement("div"); div.className="row";
    // Single-quoted JS strings so the HTML attribute double-quotes need no escaping.
    div.innerHTML='<b>'+sp.id+'</b> <audio controls src="/clip/'+sp.id+'"></audio>'+
      '<div class="txt"></div><label>Name: <input data-id="'+sp.id+'" placeholder="'+sp.id+'"></label>';
    div.querySelector(".txt").textContent='"'+sp.text+'"';
    form.appendChild(div);});
}

el("save").onclick=function(){
  var names={};
  el("names").querySelectorAll("input").forEach(function(i){names[i.dataset.id]=i.value;});
  fetch("/names",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(names)}).then(function(r){if(r.ok)loadResult();});
};

function loadResult(){
  fetch("/result").then(function(r){return r.json();}).then(function(res){
    el("preview").textContent=res.text;
    el("download").setAttribute("download",res.filename);
    show("result");});
}

function restart(){fetch("/reset",{method:"POST"}).then(function(){
  el("file").value=""; show("upload");});}
el("restart").onclick=restart; el("working-reset").onclick=restart;

show("upload");
</script></body></html>"""


def main():
    """Launch the front door: bind a free localhost port, open the browser, serve
    until interrupted (NOT block-until-submit like the CLI naming modal)."""
    app = create_app()
    server = make_server("127.0.0.1", 0, app, threaded=True)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"[tathurell] front door at {url} (opening browser; Ctrl-C to quit)...",
          file=sys.stderr)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[tathurell] shutting down", file=sys.stderr)
        server.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the structural test + full suite**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -v`
Expected: PASS (12 passed).
Run: `env -u HF_TOKEN $PY -m pytest`
Expected: full suite green, 0 failures (new webapp + progress tests added on top of the prior count).

- [ ] **Step 5: Manual smoke (not automated — browser interaction)**

Run in a REAL terminal (not backgrounded): `cd <repo root> && $PY -m tathurell.webapp`
Verify: browser opens; pick `dollop_test_a.mp3`; click Transcribe; the Working screen shows changing stage labels; the Naming screen shows two players that play; enter names; Save; the Result screen shows the transcript and Download saves `dollop_test_a.transcription.txt`; Start over returns to Upload. (Needs `HF_TOKEN` configured for the real pipeline; this is the only step that exercises the real transcriber.) Ctrl-C stops the server.

- [ ] **Step 6: Commit**

```bash
git add tathurell/webapp.py tests/test_webapp.py
git commit -m "tathurell: webapp single-page shell + python -m launcher"
```

---

## Notes for the implementer

- **Injectable transcriber is the testability spine:** `create_app(transcriber_factory=...)` lets tests pass `FakeTranscriber` so the whole flow runs with no model and no HF token in milliseconds. The default `WhisperXTranscriber` is exercised only by the manual smoke (Step 5, Task 6) and the gated `test_whisperx_core_smoke`.
- **The job runs off the request thread** (`threading.Thread`), so `/status` stays responsive while transcription runs. `_poll` in the tests waits on real work (a `FakeTranscriber` plus a real `extract_clip`).
- **One job at a time:** `/upload` is refused (409) unless `stage == "idle"`; the user returns to idle via "Start over" (`POST /reset`), which also deletes the temp dir.
- **No decode/transcription behavior changed:** the only edit to existing code is the optional `progress` callback on `transcribe` (default `None`).
- **DRY note:** the `/clip` route mirrors `naming_ui.py`'s clip serving. Phase 1 keeps the small duplication rather than refactoring `naming_ui.py` (its CLI behavior must stay intact); revisit a shared helper only if Phase 2 makes the overlap larger.
