import os
import shutil
import tempfile

from tathurell.ffmpeg import ffmpeg_exe, ensure_ffmpeg_on_path

SCRUBBED = "/nonexistent-tathurell-dir"  # a PATH with no system ffmpeg


def test_ffmpeg_exe_returns_executable():
    exe = ffmpeg_exe()
    assert os.path.isfile(exe)
    assert os.access(exe, os.X_OK)


def test_ensure_puts_bundled_ffmpeg_on_path(monkeypatch):
    # With PATH scrubbed of any system ffmpeg, a bare `ffmpeg` must resolve to
    # the bundled binary after ensure_ffmpeg_on_path() installs the shim.
    monkeypatch.setenv("PATH", SCRUBBED)
    assert shutil.which("ffmpeg") is None
    ensure_ffmpeg_on_path()
    found = shutil.which("ffmpeg")
    assert found is not None
    assert os.path.realpath(found) == os.path.realpath(ffmpeg_exe())


def test_ensure_is_idempotent(monkeypatch):
    # Calling it twice must not keep growing PATH.
    monkeypatch.setenv("PATH", SCRUBBED)
    ensure_ffmpeg_on_path()
    after_first = os.environ["PATH"]
    ensure_ffmpeg_on_path()
    assert os.environ["PATH"] == after_first


def test_ensure_replaces_stale_symlink(monkeypatch):
    # Pre-plant a wrong symlink in the shim dir; ensure_ffmpeg_on_path() must
    # replace it with one that resolves to the bundled binary.
    shim_dir = os.path.join(tempfile.gettempdir(), "tathurell_ffmpeg")
    os.makedirs(shim_dir, exist_ok=True)
    link = os.path.join(shim_dir, "ffmpeg")

    # Remove any existing link from a prior test run before planting the bogus one.
    if os.path.lexists(link):
        os.remove(link)
    os.symlink("/dev/null", link)  # bogus target — wrong binary

    monkeypatch.setenv("PATH", SCRUBBED)
    ensure_ffmpeg_on_path()

    # The stale symlink must have been replaced; it must now point to the bundled binary.
    assert os.path.islink(link), "shim link was not created"
    assert os.path.realpath(link) == os.path.realpath(ffmpeg_exe())
