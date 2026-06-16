# Front-Door UI Phase 2 — Confidence-Flagged Transcript Review — Design Spec

**Date:** 2026-06-15
**Status:** Approved for planning
**Context:** Phase 1 (the front door — upload → transcribe → name → download) is on `main`. It lets the
user *name* the speakers the diarizer found but *trusts the diarization completely*. Phase 2 adds a
review step that surfaces the diarizer's own uncertainty so the user can fix the spots it got wrong.

This spec deliberately **reshapes** the original Phase-2 wishlist ("merge / split / relabel speakers").
While scoping it we established that the existing machinery already covers most of that wishlist, so
Phase 2 narrows to the high-value, well-grounded core. See §7 for what was cut and why.

## 1. Purpose & scope

After naming, show the **whole transcript** run-by-run with speaker names; attach a **confidence** score
to each run; a client-side **slider** highlights runs below a tunable threshold; the user can **reassign
any run's speaker** (not only flagged ones) after listening to it. Reassignments relabel runs in the
final transcript.

**In scope:** per-run confidence; the slider-tuned highlighting; click-to-play any run; reassign a run
to another existing speaker; rebuild the transcript from the (possibly-edited) per-run speakers.

**Out of scope — recorded as roadmap TODOs, not built here:**
- **Duplicate-speaker merge** (embedding-based) — a genuine quick win, but punted to keep this spec on
  the meatier reassignment work. Folds naturally into the naming step later.
- **Splitting a run** when the diarizer lumped two people under one label (the under-detection case).
  Reassignment moves *whole runs*, so interleaved speakers inside one run can't be separated. Needs
  sub-run selection or segmentation posteriors — only if a real need appears.

## 2. The confidence signal (recorded findings)

Verified against the installed `whisperx/diarize.py` (whisperx 3.8.6, pyannote community-1):

**There is no native per-segment confidence field.** `DiarizationPipeline` returns hard segments
(`start, end, speaker`) — no probability. **But** `assign_word_speakers` *computes* the raw material and
discards it: for each word it builds `{speaker: overlap_duration}` over the word's time span and keeps
only the `argmax`, throwing away the margin.

**Chosen signal — overlap dominance.** Per word:

```
confidence(word) = overlap(final_speaker, word_span) / total_overlap(all_speakers, word_span)
```

where `final_speaker` is the word's speaker *after* realign. Intuition:
- word cleanly inside its speaker's turn → ~1.0 (confident),
- word straddling a turn boundary → ~0.5 (ambiguous),
- word the realign step moved to a speaker with no local diarization overlap, **or** a gap-filled
  (`fill_nearest`) word → **0.0** (pure guess; `total_overlap == 0` also yields 0).

Per **run** the score is the **minimum** of its words' confidences — one shaky word flags the run, so a
problem can't hide behind confident neighbours.

**Alternatives considered and rejected (for now):**
- **Realign vote-margin** — the per-sentence majority vote already computed in `realign.py` has a clean
  margin, but its sentence unit doesn't line up with runs. Rejected on unit mismatch.
- **pyannote segmentation posteriors** — the model's per-frame probabilities are the "true" calibrated
  confidence, but extracting them means going below the clean pipeline API (segmentation model / hooks),
  unverified for community-1. Noted as a possible later upgrade, not built.

The dominance ratio is directly computable from data the pipeline already produces and matches the
user's mental model ("the diarizer was unsure here").

## 3. Flow & screen

Slots in **after naming** so the reassignment picker shows friendly names, not `SPEAKER_xx`:

`Upload → Working → Naming → Review → Result`

```
Review who said what            confidence ▕─────●────▏  highlighting 9 of 84 runs
┌──────────────────────────────────────────────────────────────────────┐
│ Alice: the spanish version of the dollop and big fans of the show…  ▶ │
│ Bob:   welcome gentlemen                                            ▶ │
│ ⟨Bob⟩  and then we went—            (uncertain)   reassign:[Alice ▾] ▶ │  ← highlighted
│ Alice: no that's not right                                         ▶ │
└──────────────────────────────────────────────────────────────────────┘
                                                        [ Looks good ▶ ]
```

Every run is playable (▶ plays that run's span) and reassignable (a name dropdown). Flagged runs are
tinted and surface the picker. The **slider is purely client-side** — every run's confidence ships in
the page, so dragging it only re-tints; no server round-trip.

## 4. Architecture, data & endpoints

### Pipeline (one additive change)
- New pure helper module `tathurell/confidence.py`:
  `word_confidences(words, diar_segments) -> list[float]` — `diar_segments` is the list of
  `(start, end, speaker)` from the diarization dataframe; returns one dominance ratio per word
  (matching `words` order), `0.0` when total overlap is 0.
- `whisperx_core.transcribe` already holds the diarization dataframe (`diar`). After assignment +
  realign it builds `diar_segments`, computes the confidences, and attaches `word["confidence"]`.
- `naming.group_by_speaker` is extended **additively**: each run gains `start` (first word start),
  `end` (last word end), and `confidence = min(word confidences in the run)`. Existing keys
  (`speaker`, `text`) and `apply_names` are unchanged — the CLI ignores the new keys.

### Webapp (a new stage between naming and done)
Stage order becomes `… naming → review → done`.
- `POST /names` — applies names, computes the named runs (with spans + confidence), enters **review**
  (no longer goes straight to `done`).
- `GET /status` (review) — returns `{stage:"review", runs:[{i, speaker, text, start, end, confidence}],
  names:[distinct speaker names]}`.
- `GET /span/<i>` — extracts and serves run `i`'s audio on demand via `extract_clip` on `[start,end]`
  (only runs the user clicks get extracted); 404 for an out-of-range index.
- `POST /review {speakers: [name_per_run]}` — sets each run's speaker from the submitted list (applying
  any reassignments), rebuilds the final transcript, sets `done`. Length must match the run count
  (else 400).
- `GET /result`, `GET /download`, `POST /reset` — unchanged.

### SPA
The Result view from Phase 1 is preceded by a **Review** view: renders all runs, tints those below the
slider threshold, a per-run ▶ (calls `/span/<i>`) and a name `<select>`; "Looks good" POSTs the per-run
speaker list to `/review`, then loads `/result` as today.

## 5. Reassignment mechanics & boundary

Reassigning a run = setting its speaker to another existing name; `/review` rebuilds the transcript by
joining consecutive same-name runs (same render rule as `apply_names`). **Run-level only** — no text
editing, no run splitting (§1 out-of-scope). The realign fix already keeps most runs speaker-clean, so
run granularity is the right pragmatic unit.

## 6. Testing

- **Pure unit:** `word_confidences` — clean word → ~1.0; boundary word → ~0.5; gap-filled / realign-
  overridden word → 0.0; empty diarization → all 0.0. `group_by_speaker` — runs carry correct
  `start`/`end` and `confidence == min` of the run's word confidences; the CLI output (`apply_names`)
  is byte-identical to before.
- **Webapp (fake transcriber returns words *with* confidence):** the flow reaches `review` exposing
  runs + confidence + names; `GET /span/<i>` serves `audio/wav` (and 404 out-of-range); `POST /review`
  with an edited speaker list produces a reassigned, regrouped transcript at `/result`; a malformed
  list length → 400. Structural test that the Review view + per-run confidences + the slider are in the
  served page.
- **Real e2e (gated `TATHURELL_E2E`):** extend the existing webapp e2e through review — assert every run
  has a `confidence` in `[0,1]`, reassign one run, and confirm `/result` reflects it.
- Runs in the `tathurell-eval` venv; existing suite stays green.

## 7. Compatibility

- `transcribe` adds a `confidence` key per word; `group_by_speaker` adds `start`/`end`/`confidence` per
  run — both additive. The CLI (`tathurell_transcribe.py`) and its `"<name>: <text>"` output are
  unchanged.
- The webapp gains a stage; Phase 1's endpoints keep their contracts (`/names` now transitions to
  `review` instead of `done`, which only the SPA observes).
