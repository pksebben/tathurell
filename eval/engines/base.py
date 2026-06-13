"""Engine adapter contract + the shared word->speaker attribution rule.

A `Word` is {"word": str, "start": float, "end": float, "speaker": str|None}.
A `Turn` is {"speaker": str, "start": float, "end": float}.

Engines that diarize separately from ASR (vosk, mlx-whisper) call
`assign_speakers_max_overlap` to fill the speaker field. The rule: assign each
word to the diarization turn it overlaps most in time; if it overlaps no turn
(falls in a silence gap or past the last turn), assign the nearest turn by
midpoint distance. This replaces the original "first turn not yet ended"
heuristic, which ignored word end and turn start.
"""
from typing import Protocol


class Engine(Protocol):
    name: str

    def transcribe(self, audio_path: str) -> list:
        """Return a list of Word dicts with speaker filled in."""
        ...


def _overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_max_overlap(words, turns):
    for w in words:
        if not turns:
            w["speaker"] = None
            continue
        best = max(turns, key=lambda t: _overlap(w["start"], w["end"], t["start"], t["end"]))
        if _overlap(w["start"], w["end"], best["start"], best["end"]) > 0.0:
            w["speaker"] = best["speaker"]
        else:
            wm = (w["start"] + w["end"]) / 2.0
            nearest = min(turns, key=lambda t: abs(wm - (t["start"] + t["end"]) / 2.0))
            w["speaker"] = nearest["speaker"]
    return words
