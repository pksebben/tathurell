"""Run engines x corpus slices, compute WER/cpWER/DER + runtime/memory, emit a
markdown results table and save each engine's transcript for inspection.

Usage: python -m eval.run_bakeoff   (reads HF_TOKEN from env or eval/data/.hf_token)

Token loading: at startup run() checks os.environ for HF_TOKEN, then falls back
to reading eval/data/.hf_token (relative to this file's directory). If neither
source provides a token, the script exits non-zero with a clear message.

IMPORTANT: The token-loading / exit logic lives entirely inside run(), NOT at
module scope. Importing this module (e.g. in unit tests) must NOT trigger a
network call or sys.exit().
"""
from __future__ import annotations

import os
import resource
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Pure helper functions — no I/O, no models, unit-tested independently.
# ---------------------------------------------------------------------------

def words_to_hyp(words: list[dict]) -> tuple[dict[str, str], str]:
    """Group engine words by predicted speaker -> ({speaker: text}, flat_text).

    Args:
        words: list of {"word": str, "start": float, "end": float, "speaker": str}

    Returns:
        (per_spk_dict, flat_text) where per_spk_dict maps each predicted speaker
        label to its concatenated words (in word order) and flat_text is all words
        joined regardless of speaker (used for speaker-agnostic WER).
    """
    by_spk: defaultdict[str, list[str]] = defaultdict(list)
    for w in words:
        by_spk[w["speaker"]].append(w["word"])
    per_spk = {spk: " ".join(ws) for spk, ws in by_spk.items()}
    flat = " ".join(w["word"] for w in words)
    return per_spk, flat


def words_to_rttm(words: list[dict], uri: str) -> str:
    """Collapse consecutive same-speaker words into turns -> RTTM text.

    Adjacent words from the same speaker are merged into a single RTTM segment
    (the turn end is extended to the last word's end). A speaker change starts a
    new segment. Empty word lists produce an empty string (no SPEAKER lines).

    Args:
        words: list of {"word": str, "start": float, "end": float, "speaker": str}
        uri:   Recording URI used in the RTTM SPEAKER line (e.g. "ami_sdm_ES2011a").

    Returns:
        NIST RTTM text with one SPEAKER line per collapsed turn, terminated by \\n.
    """
    segs: list[dict] = []
    for w in words:
        if segs and segs[-1]["speaker"] == w["speaker"]:
            # Extend the current segment to cover this word.
            segs[-1]["end"] = w["end"]
        else:
            segs.append({
                "speaker": str(w["speaker"]),
                "start": w["start"],
                "end": w["end"],
            })
    lines = [
        f"SPEAKER {uri} 1 {s['start']:.3f} {s['end'] - s['start']:.3f}"
        f" <NA> <NA> {s['speaker']} <NA> <NA>"
        for s in segs
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# Max audio duration (seconds) for AMI and CHiME-6 excerpts. Earnings-21 runs
# its full pinned call (already short — ~18 min).
BUDGET_SECONDS = 600.0

# Path to optional token file (relative to this file's parent directory so it
# works regardless of the cwd from which the script is invoked).
_TOKEN_FILE = Path(__file__).parent / "data" / ".hf_token"


def _load_token() -> None:
    """Ensure HF_TOKEN is in os.environ, reading from file if necessary.

    Call order: env var wins; then the token file; then exit(1).
    This is intentionally a function (not module-level code) so that
    `import eval.run_bakeoff` never triggers a sys.exit().
    """
    if "HF_TOKEN" in os.environ:
        return
    if _TOKEN_FILE.exists():
        token = _TOKEN_FILE.read_text().strip()
        if token:
            os.environ["HF_TOKEN"] = token
            print(f"[bakeoff] loaded HF_TOKEN from {_TOKEN_FILE}", file=sys.stderr)
            return
    print(
        "[bakeoff] ERROR: HF_TOKEN is required but was not found.\n"
        f"  Set it in the environment:  export HF_TOKEN=hf_...\n"
        f"  Or write it to:             {_TOKEN_FILE}",
        file=sys.stderr,
    )
    sys.exit(1)


def _peak_mem_mb() -> float:
    """Return the process-cumulative peak RSS in megabytes.

    On macOS, ru_maxrss is in BYTES (unlike Linux where it is kilobytes).
    We always divide by 1024*1024 on this platform.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS: ru_maxrss in bytes.
    return usage.ru_maxrss / (1024 * 1024)


def run() -> list[dict]:
    """Run the full bake-off matrix and print a markdown results table.

    Returns the list of row dicts (corpus, engine, WER, cpWER, DER, sec, mem_MB)
    so callers (e.g. integration tests) can inspect results without parsing stdout.
    """
    # Token must be loaded before engine imports that read os.environ in __init__.
    _load_token()

    # Import corpora/metrics/engines here (after token is set).
    from eval.corpora import ami, chime6, earnings21
    from eval.metrics.cpwer import cp_wer
    from eval.metrics.der import diarization_error_rate
    from eval.metrics.wer import word_error_rate
    from eval.engines.vosk_pyannote import VoskPyannote
    from eval.engines.whisperx_stack import WhisperXStack
    from eval.engines.mlxwhisper_pyannote import MlxWhisperPyannote

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Build corpus slices.
    # ------------------------------------------------------------------
    print("[bakeoff] loading corpus slices...", file=sys.stderr)
    slices: list[dict] = []

    # AMI (far-field, excerpted for CPU tractability).
    ami_ref, ami_audio = ami.load(
        condition="sdm",
        max_seconds=BUDGET_SECONDS,
        audio_out=str(data_dir / "ami_sdm.wav"),
    )
    slices.append({
        "name": "AMI-SDM",
        "reference": ami_ref,
        "audio_path": ami_audio,
        "compute_der": True,
    })

    # Earnings-21 (telephony; no timestamps -> DER N/A).
    e21_ref, e21_audio = earnings21.load(
        audio_out=str(data_dir / "earnings21.wav"),
    )
    slices.append({
        "name": "Earnings-21",
        "reference": e21_ref,
        "audio_path": e21_audio,
        "compute_der": False,
    })

    # CHiME-6 (dinner-party far-field, excerpted for CPU tractability).
    chime_json = str(
        data_dir
        / "chime6"
        / "transcriptions"
        / "transcriptions"
        / "dev"
        / "S02.json"
    )
    chime_audio_in = str(
        data_dir
        / "chime6"
        / "CHiME6_dev"
        / "CHiME6"
        / "audio"
        / "dev"
        / "S02_U02.CH1.wav"
    )
    chime_ref, chime_audio = chime6.load(
        chime_json,
        session="S02",
        max_seconds=BUDGET_SECONDS,
        audio_in=chime_audio_in,
        audio_out=str(data_dir / "chime6_s02.wav"),
    )
    slices.append({
        "name": "CHiME-6-S02",
        "reference": chime_ref,
        "audio_path": chime_audio,
        "compute_der": True,
    })

    # ------------------------------------------------------------------
    # Instantiate engines (guarded). A failed init must NOT abort the whole
    # matrix — e.g. a whisperx/pyannote version incompatibility should still
    # leave vosk and mlx results intact. Failed engines yield ERROR rows.
    # Engines are built once and reused across all corpora (models load once).
    # ------------------------------------------------------------------
    engine_classes = [VoskPyannote, WhisperXStack, MlxWhisperPyannote]
    engines: list[tuple[str, Any]] = []
    for cls in engine_classes:
        try:
            engines.append((cls.name, cls()))
        except Exception as exc:
            print(
                f"[bakeoff] ERROR: failed to init engine {cls.name}: {exc}",
                file=sys.stderr,
            )
            engines.append((cls.name, None))

    # ------------------------------------------------------------------
    # Run matrix.
    # ------------------------------------------------------------------
    rows: list[dict] = []

    for sl in slices:
        ref = sl["reference"]
        audio = sl["audio_path"]
        corpus_name = sl["name"]
        compute_der = sl["compute_der"]
        uri = ref.uri

        # Reference flat text (for WER): join segment texts in time order.
        ref_flat = " ".join(
            s.text for s in sorted(ref.segments, key=lambda s: s.start)
        )
        ref_speaker_text = ref.per_speaker_text()

        # Write reference RTTM (only needed for DER-capable corpora).
        ref_rttm_path: str | None = None
        if compute_der:
            ref_rttm_fd, ref_rttm_path = tempfile.mkstemp(
                suffix=".rttm", prefix=f"ref_{uri}_"
            )
            try:
                with os.fdopen(ref_rttm_fd, "w") as f:
                    f.write(ref.to_rttm())
            except Exception:
                os.close(ref_rttm_fd)
                raise

        for eng_name, eng in engines:
            if eng is None:
                # Engine failed to initialize; record ERROR for every corpus.
                rows.append({
                    "corpus": corpus_name,
                    "engine": eng_name,
                    "WER": "ERROR",
                    "cpWER": "ERROR",
                    "DER": "ERROR",
                    "sec": "ERROR",
                    "mem_MB": "ERROR",
                })
                continue
            try:
                mem_before = _peak_mem_mb()
                t0 = time.perf_counter()
                words = eng.transcribe(audio)
                elapsed = time.perf_counter() - t0
                mem_after = _peak_mem_mb()
                mem_delta = mem_after - mem_before

                per_spk, flat = words_to_hyp(words)

                # Save transcript for inspection.
                transcript_path = data_dir / f"{uri}.{eng.name}.txt"
                transcript_path.write_text(flat, encoding="utf-8")

                # Compute WER and cpWER.
                wer = word_error_rate(ref_flat, flat)
                cpwer = cp_wer(reference=ref_speaker_text, hypothesis=per_spk)

                # Compute DER if timestamps are available for this corpus.
                if compute_der and ref_rttm_path is not None:
                    hyp_rttm_fd, hyp_rttm_path = tempfile.mkstemp(
                        suffix=".rttm", prefix=f"hyp_{uri}_{eng.name}_"
                    )
                    try:
                        with os.fdopen(hyp_rttm_fd, "w") as f:
                            f.write(words_to_rttm(words, uri))
                        der = diarization_error_rate(ref_rttm_path, hyp_rttm_path)
                    finally:
                        if os.path.exists(hyp_rttm_path):
                            os.unlink(hyp_rttm_path)
                else:
                    der = "N/A"

                rows.append({
                    "corpus": corpus_name,
                    "engine": eng.name,
                    "WER": wer,
                    "cpWER": cpwer,
                    "DER": der,
                    "sec": elapsed,
                    "mem_MB": mem_delta,
                })

            except Exception as exc:
                print(
                    f"[bakeoff] ERROR: {corpus_name}/{eng_name}: {exc}",
                    file=sys.stderr,
                )
                rows.append({
                    "corpus": corpus_name,
                    "engine": eng_name,
                    "WER": "ERROR",
                    "cpWER": "ERROR",
                    "DER": "ERROR",
                    "sec": "ERROR",
                    "mem_MB": "ERROR",
                })

        # Clean up reference RTTM.
        if ref_rttm_path and os.path.exists(ref_rttm_path):
            os.unlink(ref_rttm_path)

    # ------------------------------------------------------------------
    # Print markdown table.
    # ------------------------------------------------------------------
    _print_markdown_table(rows)

    return rows


def _fmt(val: object) -> str:
    """Format a metric value for the markdown table.

    Floats are shown to 3 decimal places; strings (N/A, ERROR) are passed
    through literally so the table always has readable cell content.
    """
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def _print_markdown_table(rows: list[dict]) -> None:
    headers = ["corpus", "engine", "WER", "cpWER", "DER", "sec", "mem_MB"]
    col_widths = [max(len(h), max(len(_fmt(r[h])) for r in rows)) for h in headers]

    def row_line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_widths)) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"

    print(row_line(headers))
    print(sep)
    for r in rows:
        print(row_line([_fmt(r[h]) for h in headers]))


if __name__ == "__main__":
    run()
