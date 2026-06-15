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


# Replaced with the full single-page app in a later task; kept as a constant so
# the route never changes.
_SHELL = "<!doctype html><html><body>tathurell</body></html>"


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
        job.start(tmpdir, f.filename)
        threading.Thread(
            target=_run_job, args=(job, transcriber_factory, audio_path), daemon=True
        ).start()
        return ("", 202)

    return app
