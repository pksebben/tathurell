# Bundled ffmpeg ("built-in transcoding") — Design Spec

**Date:** 2026-06-15
**Status:** Approved for planning
**Context:** The tool decodes audio in two places, and **both silently require a system-installed
`ffmpeg` on `PATH`**: `whisperx.load_audio()` shells out to a bare `ffmpeg` subprocess, and
`tathurell/sampling.py:extract_clip` uses `pydub`, which also shells out to `ffmpeg`. On a clean
machine (no `brew install ffmpeg`) the tool fails. This is a blocker for the roadmap's binary-
distribution goal, whose end user is explicitly **not** Python-proficient and will not install
ffmpeg themselves.

## 1. Purpose & scope

**Kill the dependency on a *system-installed* ffmpeg** by bundling a static ffmpeg via pip, and point
both decode consumers at it — so `pip install -r requirements.txt` (and, later, the bundled binary)
yields a tool that decodes audio on a machine with no ffmpeg on `PATH`.

**In scope:** add `imageio-ffmpeg` (static ffmpeg in-wheel); a small module that resolves the bundled
binary and puts it on `PATH`; wiring `whisperx.load_audio` (via `PATH`) and `extract_clip` (via a
direct call) to use it; dropping `pydub`.
**Out of scope (future):** the PyInstaller/briefcase binary itself (separate roadmap item); wider
input-format support or explicit normalization stages (interpretations (b)/(c) of "built-in
transcoding" that were considered and set aside — this is interpretation (a), the dependency kill).

**Decision — reliability over re-implementation.** We deliberately keep ffmpeg's exact decode rather
than decoding in-process with PyAV. ffmpeg's byte output is what the transcription pipeline is
validated against; reproducing it risks a silent quality regression at ship time. Bundling the same
ffmpeg has zero decode-fidelity risk.

## 2. Architecture

### `tathurell/ffmpeg.py` (new)
The single source of truth for "where is ffmpeg".
- `ffmpeg_exe() -> str` — return the bundled binary path from `imageio_ffmpeg.get_ffmpeg_exe()`,
  memoized. Raises a clear `RuntimeError` if the binary is missing/not executable.
- `ensure_ffmpeg_on_path() -> None` — make a bare `ffmpeg` resolve to the bundled binary,
  idempotently. **Wrinkle:** imageio-ffmpeg's binary is named like `ffmpeg-osx-arm64-v7.1`, *not*
  `ffmpeg`, so just putting its directory on PATH does **not** satisfy whisperx's hardcoded
  `subprocess.run(["ffmpeg", ...])`. So we create a stable shim dir (`<tmpdir>/tathurell_ffmpeg/`)
  holding a symlink named `ffmpeg` → the bundled binary, and prepend that dir to `os.environ["PATH"]`.
  Idempotent: returns early when `shutil.which("ffmpeg")` already realpaths to the bundled exe, and
  never double-adds the shim dir. We cannot patch whisperx, so PATH is the only injection point.
  (POSIX symlink; Windows would copy the binary instead — out of scope for the current macOS/Linux
  target.)

### `tathurell/whisperx_core.py` (change)
- In `WhisperXTranscriber.transcribe()`, call `ensure_ffmpeg_on_path()` **before**
  `whisperx.load_audio(audio_path)`. No other change — load_audio's decode is untouched, so the
  16 kHz / mono / float32 array it returns is identical to today.

### `tathurell/sampling.py` (change)
- Rewrite `extract_clip(audio_path, start, end, out_path)` to call the bundled ffmpeg **directly**
  instead of through pydub:
  ```
  ffmpeg -nostdin -y -ss <start> -i <audio_path> -t <end-start> \
         -vn -ac 2 -ar 44100 -c:a pcm_s16le <out_path>
  ```
  (`-ss` before `-i` for fast seek — cheap on long podcasts; `-vn` ignores any video/cover-art
  stream; output is 44.1 kHz stereo 16-bit PCM WAV, matching what the naming modal plays today.) Run
  via `subprocess.run(..., check=True, capture_output=True)`; on `CalledProcessError`, raise a
  `RuntimeError` carrying ffmpeg's stderr. Signature and the WAV-on-disk contract are unchanged.
- **Why drop pydub:** `imageio-ffmpeg` ships `ffmpeg` but **not** `ffprobe`, and pydub's `from_file`
  uses `ffprobe` to autodetect format. Rather than gamble on pydub tolerating a missing prober (the
  exact "works on my machine, breaks on ship" risk we are removing), we issue one ffmpeg command we
  fully control. ffmpeg autodetects the input container without needing ffprobe. `extract_clip` is
  pydub's only *runtime* use, so it leaves the runtime deps (it stays an eval-only dep — see below).

### Requirements (change)
- `requirements.txt`: **add** `imageio-ffmpeg==<pinned>` (pin captured at implementation time from the
  installed version, same convention as the rest of the file); **remove** `pydub` (runtime no longer
  imports it once `extract_clip` is rewritten).
- `eval/requirements-eval.txt`: layers on `requirements.txt`, so it inherits the bundled ffmpeg. **Add**
  `pydub` here — the eval-only vosk bake-off engine (`eval/engines/vosk_pyannote.py`) still imports it.
  Net effect: pydub moves from a runtime dep to an eval-only dep.

## 3. Data flow

```
CLI / library
  └─ WhisperXTranscriber.transcribe(path)
       ├─ ensure_ffmpeg_on_path()            # bundled ffmpeg dir → front of PATH
       └─ whisperx.load_audio(path)          # bare "ffmpeg" subprocess → resolves to bundled
  └─ naming modal
       └─ extract_clip(path, start, end, out)
            └─ subprocess[ ffmpeg_exe() ... ] # bundled ffmpeg, direct call → WAV clip
```

No system `ffmpeg` is consulted on either path once the bundled dir is at the front of `PATH`.

## 4. Tech choices

- **`imageio-ffmpeg`, not `static-ffmpeg`.** `imageio-ffmpeg` ships the static binary *inside the
  wheel* — present the instant `pip install` completes. `static-ffmpeg` *downloads* binaries on first
  use, reintroducing a network dependency at ship time (the fragility we are removing). Trade-off
  accepted: `imageio-ffmpeg` provides ffmpeg only (no ffprobe) — handled by the pydub removal above.
- **PATH injection for whisperx, direct call for extract_clip.** whisperx hardcodes `"ffmpeg"` and is
  not ours to patch, so PATH is the lever. extract_clip is ours, so we skip the indirection and call
  the binary directly — fewer moving parts, no ffprobe.
- **Keep ffmpeg's decode (reject in-process PyAV).** See §1 decision.

## 5. Error handling

- Bundled binary missing/not executable → `ffmpeg_exe()` raises `RuntimeError` with guidance
  ("reinstall imageio-ffmpeg"). Surfaces immediately, not as a cryptic decode failure.
- `extract_clip` ffmpeg failure (corrupt/unsupported input) → `RuntimeError` with ffmpeg's stderr, so
  the cause is visible. In the CLI this propagates up the existing path; the naming modal's
  `collect_names` already degrades gracefully (its caller falls back to terminal prompts on exception).
- `ensure_ffmpeg_on_path()` is idempotent — safe to call repeatedly (per transcribe) without growing
  `PATH`.

## 6. Testing

- **`ffmpeg_exe()`** returns a path that exists and is executable.
- **`ensure_ffmpeg_on_path()`** makes a bare `ffmpeg` resolve to the bundled binary: after calling it,
  `shutil.which("ffmpeg")` is non-None and `os.path.realpath` of it equals the bundled exe (it resolves
  through the `ffmpeg` symlink shim). Calling it twice does not grow `PATH`.
- **Clean-machine simulation (the load-bearing test):** with `os.environ["PATH"]` temporarily scrubbed
  of any system ffmpeg, (a) `extract_clip` on `dollop_test_a.mp3` still writes a valid ~Ns WAV, and
  (b) `whisperx.load_audio` of a short clip still decodes — proving the bundled binary, not a system
  one, does the work. (The load_audio leg may be marked slow/optional if model-free decode timing
  matters, but the subprocess decode itself needs no model.)
- **Parity:** the existing `extract_clip` tests (`tests/test_sampling_clip.py`) pass unchanged against
  the new direct-ffmpeg implementation — same WAV-duration contract.
- Runs in the `tathurell-eval` venv; existing suite stays green.

## 7. Compatibility

- Transcription output is unchanged: load_audio's decode is identical (same ffmpeg, same flags).
- `extract_clip`'s output WAV is 44.1 kHz stereo 16-bit PCM — same as the current pydub export, so the
  naming modal plays clips exactly as before.
- `pydub` removed from the **runtime** deps (its only runtime user, `extract_clip`, no longer needs it);
  it moves to `eval/requirements-eval.txt` because the eval-only vosk engine
  (`eval/engines/vosk_pyannote.py`) still imports it. No runtime code imports pydub after this change.
- On a machine that *does* have a system ffmpeg, behavior is identical — the bundled one simply takes
  precedence on `PATH`.
