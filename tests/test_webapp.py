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
