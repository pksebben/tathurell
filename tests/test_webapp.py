import os
import time

import pytest

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
    # Reassign the second run (Bob's) to Alice -> output all Alice, merged to 1 line.
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


def test_shell_wires_all_views_and_endpoints():
    html = create_app().test_client().get("/").get_data(as_text=True)
    # The four views the JS swaps between:
    for view in ("view-upload", "view-working", "view-naming", "view-result"):
        assert f'id="{view}"' in html
    # The JS talks to every endpoint:
    for ep in ("/upload", "/status", "/names", "/result", "/download", "/reset", "/clip/"):
        assert ep in html


@pytest.mark.skipif(
    not os.environ.get("TATHURELL_E2E"),
    reason="slow real-model run (~2-3 min); set TATHURELL_E2E=1 to enable. "
    "Drives the REAL WhisperX pipeline through the webapp; runs offline (no token).",
)
def test_real_pipeline_through_webapp():
    # End-to-end with the DEFAULT (real) WhisperXTranscriber: proves the actual
    # models flow through upload -> background job -> naming -> named transcript,
    # the one integration the fake transcriber cannot cover.
    app = create_app()  # real WhisperXTranscriber
    c = app.test_client()
    assert _upload(c).status_code == 202
    snap = _poll(c, "naming", tries=600, delay=1.0)  # up to ~10 min for the real run
    assert snap["stage"] == "naming"
    assert len(snap["speakers"]) >= 1
    # A real clip must be served and playable for at least the first speaker.
    first = snap["speakers"][0]["id"]
    assert c.get(f"/clip/{first}").status_code == 200
    # Name the first speaker, leave the rest to fall back to their labels.
    names = {s["id"]: ("Alice" if i == 0 else "") for i, s in enumerate(snap["speakers"])}
    assert c.post("/names", json=names).status_code == 200
    res = c.get("/result").get_json()
    assert res["filename"] == "dollop_test_a.transcription.txt"
    assert "Alice:" in res["text"]
    assert len(res["text"]) > 100
