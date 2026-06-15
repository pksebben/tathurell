# Front-Door Transcription UI — Phase 1 Design Spec

**Date:** 2026-06-15
**Status:** Approved for planning
**Context:** Today transcription is CLI-only: `tathurell_transcribe.py <audio>` runs the pipeline, then
opens the block-until-submit naming modal (`tathurell/naming_ui.py:collect_names`). The roadmap's
"fully-featured UI for the non-power-user" makes the browser the **front door** so a non-technical user
never touches a terminal. That vision is decomposed into **Phase 1 (this spec) — the walking skeleton**
(upload → transcribe-with-progress → name → preview → download) and **Phase 2 (separate, later) —
diarization management** (merge/split/relabel speakers). This spec is Phase 1 only.

## 1. Purpose & scope

A persistent local web app that drives the whole pipeline from the browser: upload an audio file, watch
coarse stage progress while it transcribes, name the detected speakers by ear (reusing today's naming),
then preview the transcript inline and download it.

**In scope:** a persistent Flask app + launcher; audio upload; a background transcription job with
coarse stage progress; reuse of the existing speaker-naming step; an inline transcript preview; a
download; "start over". One job at a time, **one-shot and stateless** (nothing persisted between runs).

**Out of scope (later):**
- **Diarization management** (merge/split/relabel) — Phase 2.
- **Persistent library / history** of past transcriptions — deliberately not built (one-shot).
- **The double-click binary** — the packaging roadmap item; Phase 1 launches via a command.
- **HF-token provisioning** — the app constructs `WhisperXTranscriber()` exactly as the CLI does and
  does not manage tokens. (A separate cleanup task is removing the now-dead token machinery; this spec
  is agnostic to it — if transcription raises for any reason, the UI shows the error.)

## 2. Architecture

A new module `tathurell/webapp.py`: a persistent Flask app, an in-process single-job state object, a
background worker thread, and a `__main__` launcher. The current `naming_ui.py:collect_names` (the CLI's
blocking modal) is unchanged and stays for the `tathurell_transcribe.py` path.

**Single-page app:** one HTML shell + vanilla JS (no framework). The JS swaps between four views by
polling/POSTing JSON endpoints. This is cleaner than server-rendered page-per-step for the progress
polling and keeps everything in one template.

**Job state (one-shot, stateless):** a single in-process object guarded by a `threading.Lock`, holding:
the per-job temp dir, the uploaded audio path, the current `stage`, an optional `error` message, the
transcription result (words + speaker groups + samples), and the final named transcript text. "Start
over" / shutdown clears it and removes the temp dir. No database, no disk persistence.

**Concurrency:** exactly one job at a time. While a job is not in the `upload` state, the UI does not
offer a new upload. Transcription runs in a **background thread**, so it never blocks the web server.
The server is started with `threaded=True` (genuine concurrency now: status polling + audio clip
requests + the worker thread).

### Screens / flow

```
UPLOAD ──POST /upload──▶ WORKING ──(poll GET /status)──▶ NAMING ──POST /names──▶ RESULT
  ▲  drag/choose file,     spinner + coarse              per-speaker            inline preview +
  │  [Transcribe]          stage label                   audio + name field     [Download] [Start over]
  └──────────────────────────────── POST /reset ◀───────────────────────────────────────┘
```

- **Upload** — drag-drop or file-picker; a Transcribe button. On submit, POST the file.
- **Working** — indeterminate spinner + the current stage label (e.g. "Identifying speakers…") + an
  upfront "this can take a few minutes for long audio" note. Driven by polling `GET /status` (~1s).
- **Naming** — one row per detected speaker: a `<audio controls>` sample clip, the sample text, and a
  name field; a "Save names" button. Reuses `pick_speaker_samples` + `extract_clip`. Blank name → the
  speaker label (same rule as today).
- **Result** — the named transcript rendered inline (scrollable), a Download button, and "Start over".

## 3. Endpoints

| Method/Path | Purpose |
|---|---|
| `GET /` | The SPA shell (HTML + JS). |
| `POST /upload` | Receive the audio file (multipart), store it in a fresh temp dir, start the background transcription job, return 202. Rejects if a job is already running or no file given. |
| `GET /status` | JSON `{stage, error?, speakers?}`. `stage` ∈ `idle` (no job yet / after reset → SPA shows Upload), `transcribing, aligning, diarizing, finishing` (job running), `naming` (awaiting names), `done`, `error`. When `stage=="naming"`, includes `speakers`: `[{id, text}]` so the JS can render the naming form (audio src `/clip/<id>`). The `/upload` POST transitions `idle → transcribing` directly. |
| `GET /clip/<speaker>` | Serve that speaker's sample WAV (same `send_file` pattern as `naming_ui.py`, with range support). 404 if the speaker isn't in the current job. |
| `POST /names` | Body `{speaker: name}`; apply names (`apply_names`), store the final transcript, set `stage=done`. |
| `GET /result` | JSON `{text, filename}` for the inline preview (filename derived from the uploaded audio). |
| `GET /download` | Serve the transcript as a `text/plain` attachment named `<audio-stem>.transcription.txt`. |
| `POST /reset` | Clear state, remove the temp dir, return to `upload`. |

## 4. Progress instrumentation

`WhisperXTranscriber.transcribe` currently runs as one opaque call. Add an optional callback:
`transcribe(self, audio_path, progress=None)`. At the stage boundaries it already has, call
`progress("transcribing")` before `model.transcribe`, `progress("aligning")` before `align`,
`progress("diarizing")` before the diarization call, and `progress("finishing")` before
`assign_word_speakers`/`realign`. **The CLI passes nothing (`progress=None`) → identical behavior, no
risk.** The webapp passes a callback that writes the stage into the shared job state for `/status`.

WhisperX exposes no finer progress, so stages are intentionally coarse (an honest indeterminate spinner,
not a fabricated percentage).

## 5. Testability (design choice)

The background job runner takes an **injectable transcriber factory** (default: `WhisperXTranscriber`).
Tests inject a fake transcriber that (a) calls the `progress` callback through the stage sequence and
(b) returns canned words — so the full upload→status→naming→download flow is exercised via the Flask
**test client** with **no model, no HF token, in milliseconds**. The real pipeline gets a single
HF-gated smoke test, skipped without a token (mirrors `tests/test_whisperx_core_smoke.py`).

## 6. Error handling

- No file / a second upload while busy → 4xx + an error shown on the Upload screen; state unchanged.
- The background job wraps the pipeline in try/except; any exception sets `stage="error"` with the
  exception message. The Working screen shows the message + a "Start over" button. The server stays
  alive (a failed job never kills the app).
- Naming submit for an unknown/again-submitted job → ignored gracefully (idempotent on the done state).
- Upload size: store streamed to the temp dir; no artificial cap for a local single-user app (note: do
  not set a restrictive `MAX_CONTENT_LENGTH`).

## 7. Reuse & file layout

- **New:** `tathurell/webapp.py` (app + job state + routes + `__main__` launcher + the SPA template
  string, matching `naming_ui.py`'s inline-HTML style).
- **Modified:** `tathurell/whisperx_core.py` — add the optional `progress` callback to `transcribe`.
- **Reused unchanged:** `pick_speaker_samples`, `extract_clip` (`sampling.py`); `group_by_speaker`,
  `apply_names` (`naming.py`); `WhisperXTranscriber` (`whisperx_core.py`); the bundled-ffmpeg activation
  already happens inside `transcribe`.
- The per-speaker clip serving overlaps with `naming_ui.py`. Phase 1 may keep a small amount of parallel
  clip-serving code in `webapp.py` rather than prematurely refactoring `naming_ui.py`; if the overlap is
  more than trivial, factor the shared sample-extraction/clip-serving into a helper both import. (Decide
  during implementation; do not restructure `naming_ui.py`'s CLI behavior.)
- Launcher: `python -m tathurell.webapp` → bind a free localhost port, open the browser, run
  `serve_forever` until interrupted (NOT block-until-submit).

## 8. Testing

- **Flask test client, fake transcriber:** `POST /upload` starts a job; `GET /status` walks through the
  stages to `naming` and exposes `speakers`; `GET /clip/<id>` serves a clip (use a tiny generated WAV in
  the fake); `POST /names` produces the expected named transcript; `GET /result` returns text + derived
  filename; `GET /download` sets the attachment filename and body; `POST /reset` clears state.
- **Progress callback:** the fake transcriber asserts the `progress` callback is invoked with the
  expected stage order; a direct unit test confirms `transcribe(progress=cb)` calls `cb` per stage
  (using a stubbed pipeline, or covered via the smoke test).
- **Busy/error paths:** second upload while running → rejected; an exception in the job → `stage=error`
  with the message; `/clip` for an unknown speaker → 404.
- **HF-gated smoke (skipped without token):** real `WhisperXTranscriber` end-to-end on
  `dollop_test_a.mp3` through the job runner.
- Runs in the `tathurell-eval` venv; existing suite stays green.

## 9. Compatibility

- The CLI (`tathurell_transcribe.py`) and `naming_ui.collect_names` are unchanged; `transcribe`'s new
  `progress` param defaults to `None`. Transcript output format (`"<name>: <text>"` lines) is identical.
- The webapp is a new, additive entry point; nothing existing depends on it.
