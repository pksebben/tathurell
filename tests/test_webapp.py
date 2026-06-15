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
