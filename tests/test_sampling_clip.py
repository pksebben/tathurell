import wave
from tathurell.sampling import extract_clip


def test_extract_clip_writes_wav_of_expected_duration(tmp_path):
    out = tmp_path / "clip.wav"
    extract_clip("dollop_test_a.mp3", 1.0, 4.0, str(out))
    assert out.exists() and out.stat().st_size > 0
    with wave.open(str(out)) as w:
        dur = w.getnframes() / w.getframerate()
    assert 2.7 < dur < 3.3  # ~3s window


def test_extract_clip_works_without_system_ffmpeg(tmp_path, monkeypatch):
    # extract_clip must not depend on a system ffmpeg: scrub PATH and confirm it
    # still produces a valid ~3s clip via the bundled binary.
    monkeypatch.setenv("PATH", "/nonexistent-tathurell-dir")
    out = tmp_path / "clip.wav"
    extract_clip("dollop_test_a.mp3", 1.0, 4.0, str(out))
    with wave.open(str(out)) as w:
        dur = w.getnframes() / w.getframerate()
    assert 2.7 < dur < 3.3
