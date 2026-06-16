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

from tathurell.naming import group_by_speaker, render_runs
from tathurell.sampling import extract_clip, pick_speaker_samples
from tathurell.whisperx_core import WhisperXTranscriber


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
        self.audio_path = None
        self.samples = None  # {speaker: {"start","end","text"}}
        self.groups = None   # [{"speaker","text"}]
        self.runs = None     # [{"speaker","text","start","end","confidence"}] for review
        self.text = None     # final named transcript

    @property
    def tmpdir(self):
        return self._tmpdir

    def start(self, tmpdir, audio_name, audio_path=None):
        with self.lock:
            self._tmpdir = tmpdir
            self.audio_name = audio_name
            self.audio_path = audio_path
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

    def set_review(self, runs):
        with self.lock:
            self.runs = runs
            self.stage = "review"

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
            if self.stage == "review" and self.runs is not None:
                snap["runs"] = [
                    {"i": i, "speaker": r["speaker"], "text": r["text"],
                     "start": r["start"], "end": r["end"], "confidence": r["confidence"]}
                    for i, r in enumerate(self.runs)
                ]
                snap["names"] = sorted({r["speaker"] for r in self.runs if r["speaker"]})
            return snap


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
    setTimeout(poll,1000);  // unknown/transient stage: keep polling rather than freeze
  }).catch(function(){setTimeout(poll,1000);});  // transient /status failure: retry
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
    body:JSON.stringify(names)}).then(function(r){
      if(r.ok){loadResult();} else {r.text().then(function(t){alert(t);});}});
};

function loadResult(){
  fetch("/result").then(function(r){return r.json();}).then(function(res){
    el("preview").textContent=res.text;
    el("download").setAttribute("download",res.filename);
    show("result");}).catch(function(){alert("Could not load the result.");});
}

function restart(){fetch("/reset",{method:"POST"}).then(function(){
  el("file").value=""; show("upload");});}
el("restart").onclick=restart; el("working-reset").onclick=restart;

show("upload");
</script></body></html>"""


def _out_name(audio_name):
    """Output filename derived from the uploaded audio name."""
    stem = os.path.splitext(os.path.basename(audio_name or "transcript"))[0]
    return f"{stem}.transcription.txt"


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

    @app.route("/upload", methods=["POST"])
    def upload():
        if job.snapshot()["stage"] != "idle":
            return ("a job is already running", 409)
        f = request.files.get("audio")
        if not f or not f.filename:
            return ("no audio file", 400)
        tmpdir = tempfile.mkdtemp(prefix="tathurell_web_")
        try:
            audio_path = os.path.join(tmpdir, "input" + os.path.splitext(f.filename)[1])
            f.save(audio_path)
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)  # don't orphan the temp dir
            raise
        job.start(tmpdir, f.filename, audio_path)
        threading.Thread(
            target=_run_job, args=(job, transcriber_factory, audio_path), daemon=True
        ).start()
        return ("", 202)

    @app.route("/clip/<speaker>")
    def clip(speaker):
        if not job.samples or speaker not in job.samples:
            return ("unknown speaker", 404)
        return send_file(os.path.join(job.tmpdir, f"{speaker}.wav"),
                         mimetype="audio/wav")

    @app.route("/span/<int:i>")
    def span(i):
        if not job.runs or i < 0 or i >= len(job.runs):
            return ("unknown run", 404)
        r = job.runs[i]
        out = os.path.join(job.tmpdir, f"span_{i}.wav")
        extract_clip(job.audio_path, r["start"], r["end"], out)
        return send_file(out, mimetype="audio/wav")

    @app.route("/names", methods=["POST"])
    def names():
        if job.groups is None or not job.samples:
            return ("no transcript to name", 409)
        data = request.get_json(silent=True) or {}
        resolved = {spk: ((data.get(spk) or "").strip() or spk) for spk in job.samples}
        named_runs = [
            {**g, "speaker": resolved.get(g["speaker"], g["speaker"])} for g in job.groups
        ]
        job.set_review(named_runs)
        return ("", 200)

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

    return app


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
