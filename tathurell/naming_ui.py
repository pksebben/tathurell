"""Local browser modal for naming speakers by ear.

create_app builds the Flask app (pure-ish, test-client friendly). collect_names
(Task 5) extracts clips, runs the app on a free port, opens the browser, and
blocks until the form is submitted.
"""
import os
import shutil
import sys
import tempfile
import threading
import webbrowser
from html import escape

from flask import Flask, request, send_file
from werkzeug.serving import make_server

from tathurell.sampling import extract_clip

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
            _ROW.format(spk=spk, text=escape(s["text"]))
            for spk, s in samples.items()
        )
        return _PAGE.format(rows=rows)

    @app.route("/clip/<speaker>")
    def clip(speaker):
        if speaker not in samples:
            return ("unknown speaker", 404)
        return send_file(os.path.join(clip_dir, f"{speaker}.wav"), mimetype="audio/wav")

    @app.route("/submit", methods=["POST"])
    def submit():
        for spk in samples:
            name = request.form.get(spk, "").strip()
            result[spk] = name or spk
        done.set()
        return "<p>Names saved. You can close this tab.</p>"

    return app


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
