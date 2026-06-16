# Front-Door UI Phase 2 — Confidence-Flagged Transcript Review — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a review step after naming where the whole transcript is shown run-by-run with a per-run confidence, a client-side slider highlights low-confidence runs, and the user can reassign any run's speaker after listening.

**Architecture:** A new pure `tathurell/confidence.py` computes each word's diarization overlap-dominance ratio; `transcribe` attaches it as `word["confidence"]`; `group_by_speaker` is extended (additively) so each run carries `start`/`end`/`confidence = min(word confidences)`. The webapp gains a `review` stage between naming and done, with `/span/<i>` (per-run audio), `/review` (finalize with reassignments), and a Review view in the SPA.

**Tech Stack:** Python 3.10 in the `tathurell-eval` venv, Flask, vanilla JS, pytest. Spec: `docs/superpowers/specs/2026-06-15-frontdoor-ui-phase2-confidence-review-design.md`.

**File structure:**
- Create `tathurell/confidence.py` — `word_confidences(words, diar_segments)`.
- Create `tests/test_confidence.py`.
- Modify `tathurell/naming.py` — extend `group_by_speaker`; add `render_runs`.
- Modify `tests/test_naming.py` — update the 4 exact-equality assertions; add enriched-run tests.
- Modify `tathurell/whisperx_core.py` — attach `word["confidence"]` in `transcribe`.
- Modify `tests/test_whisperx_core_smoke.py` — assert confidences in the gated run.
- Modify `tathurell/webapp.py` — Job review state; `/names`→review; `/span/<i>`; `/review`; SPA Review view.
- Modify `tests/test_webapp.py` — update the names flow; add review/span/reassign tests; extend the gated e2e.

> Throughout: `PY=~/.pyenv/versions/tathurell-eval/bin/python`. Run pytest as `env -u HF_TOKEN $PY -m pytest` from the repo root. Commit messages end with:
> `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

### Task 1: `tathurell/confidence.py` — per-word overlap dominance

**Files:** Create `tathurell/confidence.py`; create `tests/test_confidence.py`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_confidence.py`:

```python
from tathurell.confidence import word_confidences

# diar_segments: list of (start, end, speaker)
DIAR = [(0.0, 1.0, "A"), (1.0, 2.0, "B")]


def test_clean_word_is_confident():
    words = [{"start": 0.1, "end": 0.5, "speaker": "A"}]
    assert word_confidences(words, DIAR) == [1.0]


def test_boundary_word_is_split():
    # Word 0.5-1.5 overlaps A for 0.5 and B for 0.5; assigned to A -> 0.5.
    words = [{"start": 0.5, "end": 1.5, "speaker": "A"}]
    assert word_confidences(words, DIAR) == [0.5]


def test_word_assigned_to_speaker_with_no_overlap_is_zero():
    # Realign can move a word to a speaker that has no local diarization overlap.
    words = [{"start": 0.1, "end": 0.5, "speaker": "B"}]
    assert word_confidences(words, DIAR) == [0.0]


def test_gap_word_is_zero():
    # Word entirely outside any diarization segment (fill_nearest territory).
    words = [{"start": 5.0, "end": 5.5, "speaker": "A"}]
    assert word_confidences(words, DIAR) == [0.0]


def test_empty_diarization_is_all_zero():
    words = [{"start": 0.1, "end": 0.5, "speaker": "A"}]
    assert word_confidences(words, []) == [0.0]


def test_order_preserved_for_multiple_words():
    words = [
        {"start": 0.1, "end": 0.5, "speaker": "A"},   # 1.0
        {"start": 0.5, "end": 1.5, "speaker": "B"},   # overlaps A .5, B .5 -> B -> 0.5
    ]
    assert word_confidences(words, DIAR) == [1.0, 0.5]
```

- [ ] **Step 2: Run to verify failure**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_confidence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tathurell.confidence'`.

- [ ] **Step 3: Implement `tathurell/confidence.py`**

```python
"""Per-word diarization confidence: how cleanly a word falls under its speaker.

The diarizer emits hard segments (no probabilities), and whisperx's word->speaker
assignment discards the overlap margin. We recompute it: a word's confidence is
the fraction of its total speaker-overlap that went to its assigned (post-realign)
speaker. ~1.0 = cleanly inside one turn; ~0.5 = straddling a boundary; 0.0 = a
gap-filled word or one realign moved to a speaker with no local overlap.
"""


def word_confidences(words, diar_segments):
    """Return one confidence in [0, 1] per word, matching `words` order.

    words: [{"start","end","speaker", ...}]
    diar_segments: [(start, end, speaker), ...] from the diarization dataframe.
    """
    out = []
    for w in words:
        ws, we, spk = w["start"], w["end"], w["speaker"]
        assigned = 0.0
        total = 0.0
        for ss, se, s in diar_segments:
            overlap = min(we, se) - max(ws, ss)
            if overlap > 0:
                total += overlap
                if s == spk:
                    assigned += overlap
        out.append(assigned / total if total > 0 else 0.0)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_confidence.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add tathurell/confidence.py tests/test_confidence.py
git commit -m "tathurell: confidence.py -- per-word diarization overlap dominance"
```

---

### Task 2: Extend `group_by_speaker` (spans + min confidence) and add `render_runs`

**Files:** Modify `tathurell/naming.py`; modify `tests/test_naming.py`.

- [ ] **Step 1: Update existing assertions + add new tests** — in `tests/test_naming.py`:

First, the existing exact-equality tests now get the additive keys. Replace the bodies of the four assertions so each run dict includes `start`, `end`, `confidence`. The inputs in those tests carry no timing, so the defaults appear (`start`/`end` → `0.0`, `confidence` → `1.0`). Specifically:
- In `test_group_starts_new_run_with_triggering_word`, expect:
```python
    assert group_by_speaker(words) == [
        {"speaker": "A", "text": "hello there", "start": 0.0, "end": 0.0, "confidence": 1.0},
        {"speaker": "B", "text": "hi", "start": 0.0, "end": 0.0, "confidence": 1.0},
        {"speaker": "A", "text": "again", "start": 0.0, "end": 0.0, "confidence": 1.0},
    ]
```
- In `test_group_single_speaker_concatenates`, expect:
```python
    assert group_by_speaker(words) == [
        {"speaker": "A", "text": "a b", "start": 0.0, "end": 0.0, "confidence": 1.0}
    ]
```
- In `test_group_handles_none_speaker`, expect:
```python
    assert groups == [
        {"speaker": None, "text": "x", "start": 0.0, "end": 0.0, "confidence": 1.0},
        {"speaker": "A", "text": "y", "start": 0.0, "end": 0.0, "confidence": 1.0},
    ]
```
(The `apply_names` assertion in that test is unchanged.)
- `test_group_empty_input` and the two `apply_names` tests are unchanged.

Then ADD these new tests for the enriched behavior and `render_runs`:
```python
from tathurell.naming import render_runs


def test_group_carries_spans_and_min_confidence():
    words = [
        {"word": "a", "speaker": "A", "start": 0.0, "end": 0.5, "confidence": 0.9},
        {"word": "b", "speaker": "A", "start": 0.5, "end": 1.0, "confidence": 0.4},
        {"word": "c", "speaker": "B", "start": 1.0, "end": 1.5, "confidence": 1.0},
    ]
    groups = group_by_speaker(words)
    assert groups[0] == {
        "speaker": "A", "text": "a b", "start": 0.0, "end": 1.0, "confidence": 0.4,
    }
    assert groups[1] == {
        "speaker": "B", "text": "c", "start": 1.0, "end": 1.5, "confidence": 1.0,
    }


def test_render_runs_merges_consecutive_same_speaker():
    runs = [
        {"speaker": "Alice", "text": "hello"},
        {"speaker": "Alice", "text": "there"},
        {"speaker": "Bob", "text": "hi"},
    ]
    assert render_runs(runs) == "Alice: hello there\nBob: hi"
```

- [ ] **Step 2: Run to verify failure**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_naming.py -v`
Expected: FAIL — the updated assertions fail (keys missing) and `render_runs` import fails.

- [ ] **Step 3: Implement** — in `tathurell/naming.py`, replace `group_by_speaker` and add `render_runs`:

```python
def group_by_speaker(words):
    """Collapse consecutive same-speaker words into runs.

    Each run: {"speaker", "text", "start", "end", "confidence"} where start/end
    span the run and confidence is the MIN of the run's word confidences (one
    shaky word flags the run). start/end/confidence default to 0.0/0.0/1.0 for
    words that lack them, so callers passing bare {word,speaker} still work.
    """
    groups = []
    for w in words:
        spk = w["speaker"]
        start = w.get("start", 0.0)
        end = w.get("end", 0.0)
        conf = w.get("confidence", 1.0)
        if groups and groups[-1]["speaker"] == spk:
            g = groups[-1]
            g["text"] += f" {w['word']}"
            g["end"] = end
            g["confidence"] = min(g["confidence"], conf)
        else:
            groups.append({
                "speaker": spk, "text": w["word"],
                "start": start, "end": end, "confidence": conf,
            })
    return groups


def render_runs(runs):
    """Render runs ({"speaker","text"}) as "{speaker}: {text}" lines, merging
    consecutive runs that share a speaker (e.g. after reassignment)."""
    lines = []
    for r in runs:
        if lines and lines[-1][0] == r["speaker"]:
            lines[-1][1] += f" {r['text']}"
        else:
            lines.append([r["speaker"], r["text"]])
    return "\n".join(f"{spk}: {txt}" for spk, txt in lines)
```
(`apply_names` is unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_naming.py -v`
Expected: PASS (all naming tests).

- [ ] **Step 5: Commit**

```bash
git add tathurell/naming.py tests/test_naming.py
git commit -m "tathurell: group_by_speaker carries spans + min confidence; add render_runs"
```

---

### Task 3: Attach `confidence` to each word in `transcribe`

**Files:** Modify `tathurell/whisperx_core.py`; modify `tests/test_whisperx_core_smoke.py`.

- [ ] **Step 1: Extend the gated assertion** — in `tests/test_whisperx_core_smoke.py`, add to `test_transcriber_runs_on_clip_and_reports_progress` (after the existing assertions):

```python
    # Every word carries a diarization confidence in [0, 1].
    assert all(0.0 <= w["confidence"] <= 1.0 for w in words)
```

- [ ] **Step 2: Implement** — in `tathurell/whisperx_core.py`:

Add the import near the other `tathurell` imports:
```python
from tathurell.confidence import word_confidences
```
Then in `transcribe`, replace the final `return realign_speakers(words)` with:
```python
        # whisperx assigns each word independently, so a single word at a turn
        # boundary can flip speaker mid-sentence. Realign per sentence by majority.
        words = realign_speakers(words)
        # Attach per-word diarization confidence (overlap dominance of the final
        # speaker) so the UI can flag uncertain runs.
        diar_segments = list(zip(diar["start"], diar["end"], diar["speaker"]))
        for w, c in zip(words, word_confidences(words, diar_segments)):
            w["confidence"] = c
        return words
```

- [ ] **Step 3: Verify the fast suite still passes (gated test stays skipped)**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_whisperx_core_smoke.py -v`
Expected: `test_transcribe_accepts_progress_param` PASSES; the gated e2e SKIPPED.

- [ ] **Step 4: Verify the confidence wiring on the real pipeline (gated)**

Run: `env -u HF_TOKEN TATHURELL_E2E=1 $PY -m pytest tests/test_whisperx_core_smoke.py::test_transcriber_runs_on_clip_and_reports_progress -v`
Expected: PASS (~2-3 min, offline). This exercises the real `diar`→confidence wiring end to end.

- [ ] **Step 5: Commit**

```bash
git add tathurell/whisperx_core.py tests/test_whisperx_core_smoke.py
git commit -m "tathurell: attach per-word diarization confidence in transcribe"
```

---

### Task 4: Webapp review backend (`/names`→review, `/review` finalize)

**Files:** Modify `tathurell/webapp.py`; modify `tests/test_webapp.py`.

- [ ] **Step 1: Update the existing names test + add review tests** — in `tests/test_webapp.py`:

Replace `test_names_then_result_and_download` with the new two-step flow (names → review → finalize), and add review/reassign/validation tests:
```python
def test_names_then_review_then_result_and_download():
    app = create_app(transcriber_factory=FakeTranscriber)
    c = app.test_client()
    _upload(c)
    _poll(c, "naming")
    # Name speakers; leaving SPEAKER_01 blank -> falls back to its label.
    assert c.post("/names", json={"SPEAKER_00": "Alice", "SPEAKER_01": ""}).status_code == 200

    snap = _poll(c, "review")
    assert snap["stage"] == "review"
    runs = snap["runs"]
    assert [r["speaker"] for r in runs] == ["Alice", "SPEAKER_01"]
    assert all("confidence" in r and "start" in r and "end" in r for r in runs)
    assert snap["names"] == ["Alice", "SPEAKER_01"]

    # Finalize unchanged: keep each run's current speaker.
    speakers = [r["speaker"] for r in runs]
    assert c.post("/review", json={"speakers": speakers}).status_code == 200
    assert c.get("/status").get_json()["stage"] == "done"

    res = c.get("/result").get_json()
    assert res["filename"] == "dollop_test_a.transcription.txt"
    assert "Alice: the spanish version" in res["text"]
    assert "SPEAKER_01: welcome gentlemen" in res["text"]

    dl = c.get("/download")
    assert "dollop_test_a.transcription.txt" in dl.headers["Content-Disposition"]


def test_review_reassignment_relabels_and_merges():
    app = create_app(transcriber_factory=FakeTranscriber)
    c = app.test_client()
    _upload(c)
    _poll(c, "naming")
    c.post("/names", json={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
    snap = _poll(c, "review")
    # Reassign the second run (Bob's) to Alice -> output should be all Alice,
    # merged into a single line.
    assert c.post("/review", json={"speakers": ["Alice", "Alice"]}).status_code == 200
    text = c.get("/result").get_json()["text"]
    assert "Bob" not in text
    assert text.count("Alice:") == 1  # consecutive same-speaker runs merged


def test_review_rejects_wrong_length():
    app = create_app(transcriber_factory=FakeTranscriber)
    c = app.test_client()
    _upload(c)
    _poll(c, "naming")
    c.post("/names", json={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
    _poll(c, "review")
    assert c.post("/review", json={"speakers": ["Alice"]}).status_code == 400
```

- [ ] **Step 2: Run to verify failure**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -k "review or names_then" -v`
Expected: FAIL — `/names` still sets `done` (no `review` stage), `/review` route missing.

- [ ] **Step 3: Implement** — in `tathurell/webapp.py`:

(a) Import `render_runs`: change the naming import line to
```python
from tathurell.naming import apply_names, group_by_speaker, render_runs
```
(b) `Job`: add `audio_path` + `runs` to state and a `set_review`. In `_init_state` add:
```python
        self.audio_path = None
        self.runs = None     # [{"speaker","text","start","end","confidence"}] for review
```
Change `start` to accept an optional audio path:
```python
    def start(self, tmpdir, audio_name, audio_path=None):
        with self.lock:
            self._tmpdir = tmpdir
            self.audio_name = audio_name
            self.audio_path = audio_path
            self.error = None
            self.stage = "transcribing"
```
Add:
```python
    def set_review(self, runs):
        with self.lock:
            self.runs = runs
            self.stage = "review"
```
Extend `snapshot` — after the `naming` block, add:
```python
            if self.stage == "review" and self.runs is not None:
                snap["runs"] = [
                    {"i": i, "speaker": r["speaker"], "text": r["text"],
                     "start": r["start"], "end": r["end"], "confidence": r["confidence"]}
                    for i, r in enumerate(self.runs)
                ]
                snap["names"] = sorted({r["speaker"] for r in self.runs if r["speaker"]})
```
(c) `/upload`: pass the audio path to `start` — change `job.start(tmpdir, f.filename)` to:
```python
        job.start(tmpdir, f.filename, audio_path)
```
(d) `/names`: route to review instead of done. Replace the body after the guard with:
```python
        data = request.get_json(silent=True) or {}
        resolved = {spk: ((data.get(spk) or "").strip() or spk) for spk in job.samples}
        named_runs = [
            {**g, "speaker": resolved.get(g["speaker"], g["speaker"])} for g in job.groups
        ]
        job.set_review(named_runs)
        return ("", 200)
```
(e) Add the `/review` route (inside `create_app`, near `/names`):
```python
    @app.route("/review", methods=["POST"])
    def review():
        if job.runs is None:
            return ("not in review", 409)
        speakers = (request.get_json(silent=True) or {}).get("speakers")
        if not isinstance(speakers, list) or len(speakers) != len(job.runs):
            return ("speakers length mismatch", 400)
        runs = [{**r, "speaker": (s or r["speaker"])} for r, s in zip(job.runs, speakers)]
        job.set_done(render_runs(runs))
        return ("", 200)
```

- [ ] **Step 4: Run to verify pass**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -v`
Expected: PASS (all webapp tests, including the new review ones).

- [ ] **Step 5: Commit**

```bash
git add tathurell/webapp.py tests/test_webapp.py
git commit -m "tathurell: webapp review stage (/names->review, /review finalize)"
```

---

### Task 5: Webapp `/span/<i>` — per-run audio on demand

**Files:** Modify `tathurell/webapp.py`; modify `tests/test_webapp.py`.

- [ ] **Step 1: Write the failing test** — add to `tests/test_webapp.py`:

```python
def test_span_serves_run_audio_and_404():
    app = create_app(transcriber_factory=FakeTranscriber)
    c = app.test_client()
    _upload(c)
    _poll(c, "naming")
    c.post("/names", json={"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
    _poll(c, "review")
    r = c.get("/span/0")
    assert r.status_code == 200
    assert r.mimetype == "audio/wav"
    assert c.get("/span/99").status_code == 404
```

- [ ] **Step 2: Run to verify failure**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py::test_span_serves_run_audio_and_404 -v`
Expected: FAIL — no `/span` route (404 for index 0 too).

- [ ] **Step 3: Implement** — add the route inside `create_app` (near `/clip`):

```python
    @app.route("/span/<int:i>")
    def span(i):
        if not job.runs or i < 0 or i >= len(job.runs):
            return ("unknown run", 404)
        r = job.runs[i]
        out = os.path.join(job.tmpdir, f"span_{i}.wav")
        extract_clip(job.audio_path, r["start"], r["end"], out)
        return send_file(out, mimetype="audio/wav")
```

- [ ] **Step 4: Run to verify pass**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -v`
Expected: PASS (all webapp tests).

- [ ] **Step 5: Commit**

```bash
git add tathurell/webapp.py tests/test_webapp.py
git commit -m "tathurell: webapp /span/<i> serves a run's audio for review"
```

---

### Task 6: SPA Review view (transcript + slider + reassignment)

**Files:** Modify `tathurell/webapp.py` (the `_SHELL` string + the JS); modify `tests/test_webapp.py`.

- [ ] **Step 1: Write the failing structural test** — update `test_shell_wires_all_views_and_endpoints` in `tests/test_webapp.py` to also require the review view + slider + endpoints:

```python
def test_shell_wires_all_views_and_endpoints():
    html = create_app().test_client().get("/").get_data(as_text=True)
    for view in ("view-upload", "view-working", "view-naming", "view-review", "view-result"):
        assert f'id="{view}"' in html
    for ep in ("/upload", "/status", "/names", "/review", "/span/", "/result", "/download", "/reset", "/clip/"):
        assert ep in html
    assert 'id="conf-slider"' in html
```

- [ ] **Step 2: Run to verify failure**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py::test_shell_wires_all_views_and_endpoints -v`
Expected: FAIL — no `view-review` / `conf-slider` / `/review` / `/span/` in the shell yet.

- [ ] **Step 3: Implement** — in `tathurell/webapp.py`'s `_SHELL`:

(a) Add the Review view div, immediately BEFORE the `<div id="view-result"...>` div:
```html
<div id="view-review" class="hidden">
  <h3>Review who said what</h3>
  <p class="txt">Uncertain spans are highlighted. Drag to flag more or fewer; click ▶ to listen,
     and change a speaker if it's wrong.</p>
  <label>confidence <input type="range" id="conf-slider" min="0" max="100" value="40"></label>
  <span id="flag-count"></span>
  <div id="runs"></div>
  <button id="review-save">Looks good</button>
</div>
```
(b) In the `poll()` function, change the `done` branch to route to review first. Replace:
```js
    if(s.stage==="naming"){renderNaming(s.speakers); show("naming");return;}
    if(s.stage==="done"){loadResult();return;}
```
with:
```js
    if(s.stage==="naming"){renderNaming(s.speakers); show("naming");return;}
    if(s.stage==="review"){renderReview(s.runs, s.names); show("review");return;}
    if(s.stage==="done"){loadResult();return;}
```
(c) Change the naming "Save names" handler so that after a 200 it resumes polling (to pick up the new `review` stage) instead of calling `loadResult()` directly. Replace the `el("save").onclick` body's success branch:
```js
      if(r.ok){loadResult();} else {r.text().then(function(t){alert(t);});}
```
with:
```js
      if(r.ok){poll();} else {r.text().then(function(t){alert(t);});}
```
(d) Add the review rendering + slider + save logic. Insert this JS just before the final `show("upload");` line:
```js
var REVIEW={runs:[],names:[]};
function tint(){
  var thr=parseInt(el("conf-slider").value,10)/100, n=0;
  el("runs").querySelectorAll(".run").forEach(function(d){
    var low=parseFloat(d.dataset.conf)<thr;
    d.style.background=low?"#fff3cd":""; if(low)n++;});
  el("flag-count").textContent="highlighting "+n+" of "+REVIEW.runs.length+" runs";
}
function renderReview(runs,names){
  REVIEW.runs=runs; REVIEW.names=names;
  var box=el("runs"); box.innerHTML="";
  runs.forEach(function(r){
    var d=document.createElement("div"); d.className="run"; d.dataset.conf=r.confidence;
    var opts=names.map(function(nm){
      return '<option'+(nm===r.speaker?' selected':'')+'>'+nm+'</option>';}).join("");
    d.innerHTML='<select data-i="'+r.i+'">'+opts+'</select> '+
      '<button data-span="'+r.i+'">▶</button> '+
      '<span class="rtext"></span>';
    d.querySelector(".rtext").textContent=r.text;
    box.appendChild(d);
  });
  box.querySelectorAll("button[data-span]").forEach(function(b){
    b.onclick=function(){new Audio("/span/"+b.dataset.span).play();};});
  tint();
}
el("conf-slider").oninput=tint;
el("review-save").onclick=function(){
  var sel=el("runs").querySelectorAll("select"), speakers=[];
  sel.forEach(function(s){speakers[parseInt(s.dataset.i,10)]=s.value;});
  fetch("/review",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({speakers:speakers})}).then(function(r){
      if(r.ok){loadResult();} else {r.text().then(function(t){alert(t);});}});
};
```
(e) Add `"review"` to the `views` array near the top of the script so `show()` hides it correctly. Replace:
```js
var views=["upload","working","naming","result"];
```
with:
```js
var views=["upload","working","naming","review","result"];
```

- [ ] **Step 4: Run the structural test + full suite**

Run: `env -u HF_TOKEN $PY -m pytest tests/test_webapp.py -v`
Expected: PASS (all webapp tests, including the updated structural test).
Run: `env -u HF_TOKEN $PY -m pytest -q`
Expected: full suite green, 0 failures.

- [ ] **Step 5: Manual smoke (not automated — browser)**

Run in a REAL terminal: `$PY -m tathurell.webapp`. Upload `dollop_test_a.mp3`, name the speakers, and confirm the Review screen shows the transcript with highlighted runs, the slider re-tints, ▶ plays a span, changing a dropdown + "Looks good" produces a reassigned transcript. (Needs the offline models — they're cached.) Ctrl-C to stop.

- [ ] **Step 6: Commit**

```bash
git add tathurell/webapp.py tests/test_webapp.py
git commit -m "tathurell: webapp Review view -- transcript, confidence slider, reassignment"
```

---

### Task 7: Extend the gated real e2e through review

**Files:** Modify `tests/test_webapp.py`.

- [ ] **Step 1: Extend the e2e** — replace `test_real_pipeline_through_webapp`'s tail (from the `/names` POST onward) so it goes through review:

```python
    assert c.post("/names", json=names).status_code == 200
    snap = _poll(c, "review", tries=600, delay=1.0)
    assert snap["stage"] == "review"
    runs = snap["runs"]
    assert len(runs) >= 1
    assert all(0.0 <= r["confidence"] <= 1.0 for r in runs)
    # Reassign the first run to a fixed name, keep the rest, finalize.
    speakers = [r["speaker"] for r in runs]
    speakers[0] = "Alice"
    assert c.post("/review", json={"speakers": speakers}).status_code == 200
    res = c.get("/result").get_json()
    assert res["filename"] == "dollop_test_a.transcription.txt"
    assert "Alice:" in res["text"]
    assert len(res["text"]) > 100
```
(The setup — `_upload`, `_poll(c, "naming")`, building `names` from the naming speakers — stays as it is; only the post-naming tail changes. Keep the existing `names = {...}` construction that names the first speaker "Alice".)

- [ ] **Step 2: Run the gated e2e**

Run: `env -u HF_TOKEN TATHURELL_E2E=1 $PY -m pytest tests/test_webapp.py::test_real_pipeline_through_webapp -v`
Expected: PASS (~2-3 min, offline) — the real pipeline flows through naming → review (with real confidences) → reassignment → result.

- [ ] **Step 3: Commit**

```bash
git add tests/test_webapp.py
git commit -m "tathurell: extend real-pipeline e2e through the review stage"
```

---

## Notes for the implementer

- **Confidence is min, not mean:** one shaky word flags a run (spec §2). The min lives in `group_by_speaker`; `word_confidences` is per-word.
- **`/names` now goes to `review`, not `done`** — this is the one Phase 1 contract change; only the SPA observes it. The names test was updated to the two-step flow in Task 4.
- **The slider is client-side only** — confidences ship in `/status`; dragging re-tints via `data-conf`, no server call. `/review` is the only write.
- **Reassignment is run-level** — no splitting a run; the design (spec §1/§5) punts that. `render_runs` merges consecutive same-speaker runs so a reassignment that creates adjacent same-name runs collapses cleanly.
- **`/span/<i>` re-extracts on demand** from `job.audio_path` (stored at upload) — only runs the user clicks get a clip.
