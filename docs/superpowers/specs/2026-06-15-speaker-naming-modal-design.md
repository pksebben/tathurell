# Speaker-Naming Modal — Design Spec

**Date:** 2026-06-15
**Status:** Approved for planning
**Context:** Replaces the terminal `input()` speaker-naming loop in `tathurell_transcribe.py`. The
diarization realign fix (`tathurell/realign.py`) already cleaned up the fragmented samples that made
the old text-only prompt hard to use; this adds the missing capability — *hearing* each speaker.

## 1. Purpose & scope

Let the user name speakers **by ear**: after transcription, open a local browser modal that plays an
~8-second audio sample of each detected speaker, with a name field per speaker; submitting writes the
named transcript. Replaces the per-speaker `input("name:")` prompts.

**In scope:** sample selection + clip extraction, a local web modal (one row per speaker), wiring it
into the CLI, and a non-interactive fallback.
**Out of scope (future):** persistence / recognizing the same speaker across recordings (that needs
voice fingerprinting — a separate, larger effort). This is single-session naming.

## 2. Architecture

Three pieces, building on the existing `tathurell/` package.

### `tathurell/sampling.py`
- `pick_speaker_samples(words, max_seconds=8.0) -> dict[str, dict]` — **pure.** For each distinct
  speaker, find their **longest contiguous run** of consecutive same-speaker words; return
  `{speaker: {"start": float, "end": float, "text": str}}` where `start` is the run's first word
  start, `end` is `min(run_end, start + max_seconds)`, and `text` is the run's words joined (only the
  words within the capped window). Speakers are keyed by their label. Input is our standard word list
  (`{word, start, end, speaker}`); a word whose `speaker` is `None` is ignored for sample selection.
- `extract_clip(audio_path, start, end, out_path) -> None` — I/O. Write the `[start, end]` slice of
  `audio_path` to `out_path` as a WAV (via `soundfile`, reading only the needed frames, as the corpus
  loaders do).

### `tathurell/naming_ui.py`
- `collect_names(samples, audio_path, open_browser=True) -> dict[str, str]` — the local web app.
  1. Extract one clip per speaker (`extract_clip`) into a temp dir.
  2. Start a **Flask** app on `127.0.0.1:<free port>` (bind port 0 to get a free one) on a background
     thread.
  3. `webbrowser.open` the page (unless `open_browser=False`).
  4. **Block the calling thread until the form is submitted** (a `threading.Event` set by the POST
     handler), then shut the server down and return `{speaker: name}`.
  5. Clean up the temp clip dir.
- Routes:
  - `GET /` — render the modal: one row per speaker with an HTML5 `<audio controls>` whose `src` is
    `/clip/<speaker>`, the sample text, and a `<input name="<speaker>">` name field; a Submit button
    POSTs the form.
  - `GET /clip/<speaker>` — serve that speaker's WAV clip.
  - `POST /submit` — read the form into `{speaker: name}` (empty name → fall back to the speaker
    label), store it, set the done-event, return a "you can close this tab" page.

### `tathurell_transcribe.py` (CLI) changes
After `group_by_speaker(words)`:
- If naming UI is enabled (default) **and** usable: `samples = pick_speaker_samples(words)`;
  `names = collect_names(samples, audio_path)`.
- Else (fallback): `names = prompt_names(groups)` (the existing `input()` loop, kept).
- Then `apply_names(groups, names)` and write as today.
- New flag `--no-ui` forces the fallback. If the browser/server can't start, log a warning and fall
  back automatically.

## 3. Data flow

```
transcribe → realign → words ─┬→ group_by_speaker ───────────────┐
                              └→ pick_speaker_samples → extract clips → modal (play/name/submit) → names ┘
                                                                          → apply_names(groups, names) → write <audio>.transcription.txt
```

## 4. Tech choices

- **Flask** for the modal — a one-page form with block-until-submit is small and clean. Served on a
  free localhost port, background thread, shut down after submit. (Rationale over stdlib
  `http.server`: clearer routing/shutdown; over FastAPI: no async ceremony for a blocking one-pager.)
  Flask is added to `eval/requirements-eval.txt` / the project deps.
- Clips: temp dir (`tempfile.mkdtemp`), served as static files, removed after `collect_names` returns.
- Browser launch: `webbrowser.open`.

## 5. Error handling & fallback

- `--no-ui`, or `webbrowser`/server failing to start → fall back to `prompt_names` (the `input()`
  loop) with a stderr note. So the tool still works headless/CI.
- Empty name submitted for a speaker → use the speaker label (same rule as the current EOF fallback).
- User closes the tab without submitting → the CLI is blocked on the done-event; **Ctrl-C** aborts and
  falls back to labels. (No silent hang beyond that — documented in the "close the tab" page text.)
- Port in use → avoided by binding port 0 (OS-assigned free port).
- A speaker with words always has at least one run, so every speaker gets a sample; a speaker whose
  only words are `None`-speaker is, by definition, not a speaker key.

## 6. Testing

- **Pure unit tests** — `pick_speaker_samples`: picks the longest run (not the first), caps to
  `max_seconds`, returns correct text/span, handles a single speaker and ignores `None`-speaker words.
- **Submit handler** — Flask **test client**: `POST /submit` with form data → correct `{speaker:name}`
  dict; empty field → label fallback. `GET /` renders a row/input per speaker. No browser needed.
- **Clip extraction** — smoke test: `extract_clip` on `dollop_test_a.mp3` writes a WAV whose duration
  ≈ requested window.
- **Full browser click** — manual (can't automate the click); the pieces above cover the logic.
- Runs in the `tathurell-eval` venv. Existing suite stays green.

## 7. Compatibility

Output format (`"{name}: {text}"` lines → `<audio>.transcription.txt`) is unchanged. The `--no-ui`
path preserves today's exact behavior. The positional `audio_path` CLI form still works; `--no-ui`
joins `--output`/`--model`.
