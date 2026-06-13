"""Common ground-truth reference format shared by all corpus loaders.

Corpus-native formats (AMI NXT XML, Earnings-21 token files, CHiME-6 JSON) are
converted into `Reference` here so engine adapters and metrics never touch a
corpus-specific schema. One reference per recording (uri).
"""
from dataclasses import dataclass, field


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class Reference:
    uri: str
    segments: list = field(default_factory=list)

    def per_speaker_text(self):
        """{speaker: concatenated text} in time order — input for cpWER reference."""
        ordered = sorted(self.segments, key=lambda s: s.start)
        out = {}
        for s in ordered:
            t = s.text.strip()
            if not t:
                continue
            out[s.speaker] = f"{out[s.speaker]} {t}" if s.speaker in out else t
        return out

    def to_rttm(self):
        """NIST RTTM text (one SPEAKER line per segment) — input for DER."""
        lines = []
        for s in sorted(self.segments, key=lambda s: s.start):
            dur = s.end - s.start
            lines.append(
                f"SPEAKER {self.uri} 1 {s.start:.3f} {dur:.3f} <NA> <NA> {s.speaker} <NA> <NA>"
            )
        return "\n".join(lines) + "\n"
