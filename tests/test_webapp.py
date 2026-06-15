import time
import wave

from tathurell.webapp import Job, create_app


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
    snap = None
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
    _upload(c)
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


def test_shell_wires_all_views_and_endpoints():
    html = create_app().test_client().get("/").get_data(as_text=True)
    # The four views the JS swaps between:
    for view in ("view-upload", "view-working", "view-naming", "view-result"):
        assert f'id="{view}"' in html
    # The JS talks to every endpoint:
    for ep in ("/upload", "/status", "/names", "/result", "/download", "/reset", "/clip/"):
        assert ep in html
