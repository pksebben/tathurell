"""Pick a representative audio sample per speaker, and extract the clip.

pick_speaker_samples is pure (operates on the word list). extract_clip does I/O
via the bundled ffmpeg binary (tathurell.ffmpeg) — no system ffmpeg required.
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


def extract_clip(audio_path, start, end, out_path):
    """Write the [start, end] second slice of audio_path to out_path as a WAV.

    Calls the bundled ffmpeg directly (tathurell.ffmpeg) so no system ffmpeg is
    needed. Output is 44.1 kHz stereo 16-bit PCM, which the naming modal plays
    in-browser. `-ss` before `-i` is a fast seek (cheap on long files); `-vn`
    drops any video/cover-art stream.
    """
    import subprocess

    from tathurell.ffmpeg import ffmpeg_exe

    cmd = [
        ffmpeg_exe(), "-nostdin", "-y",
        "-ss", str(start), "-i", audio_path, "-t", str(end - start),
        "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to extract clip from {audio_path!r}: "
            f"{proc.stderr.decode(errors='replace')}"
        )
