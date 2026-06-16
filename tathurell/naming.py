"""Pure grouping + naming for diarized word lists. No models, no I/O.

A word is {"word": str, "speaker": str | None, ...}. group_by_speaker collapses
consecutive same-speaker words into runs; the word that triggers a speaker change
starts the new run (the bug the original code had: it dropped that word).
"""


def group_by_speaker(words):
    """Collapse consecutive same-speaker words into runs.

    Each run: {"speaker", "text", "start", "end", "confidence"} where start/end
    span the run and confidence is the MIN of the run's word confidences (one
    shaky word flags the run). start/end/confidence default to 0.0/0.0/1.0 for
    words that lack them, so callers passing bare {word,speaker} still work.
    """
    groups = []
    for w in words:
        spk = w["speaker"]
        start = w.get("start", 0.0)
        end = w.get("end", 0.0)
        conf = w.get("confidence", 1.0)
        if groups and groups[-1]["speaker"] == spk:
            g = groups[-1]
            g["text"] += f" {w['word']}"
            g["end"] = end
            g["confidence"] = min(g["confidence"], conf)
        else:
            groups.append({
                "speaker": spk, "text": w["word"],
                "start": start, "end": end, "confidence": conf,
            })
    return groups


def render_runs(runs):
    """Render runs ({"speaker","text"}) as "{speaker}: {text}" lines, merging
    consecutive runs that share a speaker (e.g. after reassignment)."""
    lines = []
    for r in runs:
        if lines and lines[-1][0] == r["speaker"]:
            lines[-1][1] += f" {r['text']}"
        else:
            lines.append([r["speaker"], r["text"]])
    return "\n".join(f"{spk}: {txt}" for spk, txt in lines)


def apply_names(groups, names):
    """Render runs as "{name}: {text}" lines. Unmapped speakers use their label."""
    return "\n".join(
        f"{names.get(g['speaker'], g['speaker'])}: {g['text']}" for g in groups
    )
