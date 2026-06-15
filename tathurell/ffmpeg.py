"""Resolve and activate the bundled ffmpeg, so the tool needs no system ffmpeg.

whisperx.load_audio() and tathurell.sampling.extract_clip both decode via ffmpeg.
imageio-ffmpeg ships a static ffmpeg inside its wheel; this module exposes that
binary and makes a bare `ffmpeg` on PATH resolve to it (whisperx hardcodes a
`subprocess.run(["ffmpeg", ...])` we cannot patch).
"""
import os
import shutil
import tempfile

import imageio_ffmpeg

_exe = None  # memoized bundled-binary path


def ffmpeg_exe():
    """Absolute path to the bundled ffmpeg binary (memoized)."""
    global _exe
    if _exe is None:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if not (exe and os.path.isfile(exe) and os.access(exe, os.X_OK)):
            raise RuntimeError(
                f"bundled ffmpeg missing or not executable at {exe!r}; "
                "reinstall imageio-ffmpeg"
            )
        _exe = exe
    return _exe


def ensure_ffmpeg_on_path():
    """Make a bare `ffmpeg` resolve to the bundled binary. Idempotent.

    imageio-ffmpeg's binary is named like `ffmpeg-osx-arm64-v7.1`, not `ffmpeg`,
    so putting its directory on PATH is not enough for whisperx's hardcoded
    `subprocess.run(["ffmpeg", ...])`. We keep a stable shim dir holding a symlink
    named `ffmpeg` -> the bundled binary, and prepend it to PATH.
    """
    exe = ffmpeg_exe()
    current = shutil.which("ffmpeg")
    # If PATH already resolves a bare `ffmpeg` to the bundled binary (we've run
    # before, or the shim dir is already on PATH), there's nothing to do.
    if current and os.path.realpath(current) == os.path.realpath(exe):
        return

    shim_dir = os.path.join(tempfile.gettempdir(), "tathurell_ffmpeg")
    os.makedirs(shim_dir, exist_ok=True)
    link = os.path.join(shim_dir, "ffmpeg")
    if not (os.path.islink(link) and os.path.realpath(link) == os.path.realpath(exe)):
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(exe, link)  # POSIX; Windows would copy instead (out of scope)

    parts = os.environ.get("PATH", "").split(os.pathsep)
    if shim_dir not in parts:
        os.environ["PATH"] = shim_dir + os.pathsep + os.environ.get("PATH", "")
