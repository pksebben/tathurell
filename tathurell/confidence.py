"""Per-word diarization confidence: how cleanly a word falls under its speaker.

The diarizer emits hard segments (no probabilities), and whisperx's word->speaker
assignment discards the overlap margin. We recompute it: a word's confidence is
the fraction of its total speaker-overlap that went to its assigned (post-realign)
speaker. ~1.0 = cleanly inside one turn; ~0.5 = straddling a boundary; 0.0 = a
gap-filled word or one realign moved to a speaker with no local overlap.
"""


def word_confidences(words, diar_segments):
    """Return one confidence in [0, 1] per word, matching `words` order.

    words: [{"start","end","speaker", ...}]
    diar_segments: [(start, end, speaker), ...] from the diarization dataframe.
    """
    out = []
    for w in words:
        ws, we, spk = w["start"], w["end"], w["speaker"]
        assigned = 0.0
        total = 0.0
        for ss, se, s in diar_segments:
            overlap = min(we, se) - max(ws, ss)
            if overlap > 0:
                total += overlap
                if s == spk:
                    assigned += overlap
        out.append(assigned / total if total > 0 else 0.0)
    return out
