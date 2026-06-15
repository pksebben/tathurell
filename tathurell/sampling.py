"""Pick a representative audio sample per speaker, and extract the clip.

pick_speaker_samples is pure (operates on the word list). extract_clip does I/O.
"""


def pick_speaker_samples(words, max_seconds=8.0):
    """For each speaker, return their longest contiguous run as a sample window.

    Returns {speaker: {"start": float, "end": float, "text": str}}. The window is
    the longest run's first-word start to either the run end or start+max_seconds,
    whichever is sooner; text is the run's words whose start falls in that window.
    Words with speaker None are ignored (not a nameable speaker).
    """
    runs = []  # list of (speaker, [word, ...])
    for w in words:
        spk = w.get("speaker")
        if spk is None:
            continue
        if runs and runs[-1][0] == spk:
            runs[-1][1].append(w)
        else:
            runs.append((spk, [w]))

    best = {}  # speaker -> (duration, run_words)
    for spk, ws in runs:
        dur = ws[-1]["end"] - ws[0]["start"]
        if spk not in best or dur > best[spk][0]:
            best[spk] = (dur, ws)

    out = {}
    for spk, (_dur, ws) in best.items():
        start = ws[0]["start"]
        cap_end = start + max_seconds
        end = min(ws[-1]["end"], cap_end)
        text = " ".join(w["word"] for w in ws if w["start"] < cap_end)
        out[spk] = {"start": start, "end": end, "text": text}
    return out
