"""Pure grouping + naming for diarized word lists. No models, no I/O.

A word is {"word": str, "speaker": str | None, ...}. group_by_speaker collapses
consecutive same-speaker words into runs; the word that triggers a speaker change
starts the new run (the bug the original code had: it dropped that word).
"""


def group_by_speaker(words):
    """Collapse consecutive same-speaker words into [{"speaker", "text"}] runs."""
    groups = []
    for w in words:
        spk = w["speaker"]
        if groups and groups[-1]["speaker"] == spk:
            groups[-1]["text"] += f" {w['word']}"
        else:
            groups.append({"speaker": spk, "text": w["word"]})
    return groups


def apply_names(groups, names):
    """Render runs as "{name}: {text}" lines. Unmapped speakers use their label."""
    return "\n".join(
        f"{names.get(g['speaker'], g['speaker'])}: {g['text']}" for g in groups
    )
