"""mlx-whisper (Apple-Silicon-native, large-v3) ASR + pyannote 4.x diarization +
max-overlap attribution. The speed contender on this Mac.

Diarization mirrors vosk_pyannote.py exactly — both adapters share the same
pyannote 4.x `token=` convention and `itertracks(yield_label=True)` pattern
so that a single token fix touches both consistently.
"""
import os

import mlx_whisper
import torch
import torchaudio
from pyannote.audio import Pipeline

from eval.engines.base import assign_speakers_max_overlap

MLX_REPO = "mlx-community/whisper-large-v3-mlx"


class MlxWhisperPyannote:
    name = "mlxwhisper_pyannote"

    def __init__(self):
        # pyannote 4.x renamed use_auth_token -> token (mirrors vosk_pyannote.py)
        self._dia = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=os.environ["HF_TOKEN"]
        ).to(torch.device("mps"))

    def _diarize(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        dia = self._dia({"waveform": waveform, "sample_rate": sr})
        return [
            {"speaker": spk, "start": turn.start, "end": turn.end}
            for turn, _, spk in dia.itertracks(yield_label=True)
        ]

    def transcribe(self, audio_path):
        out = mlx_whisper.transcribe(
            audio_path, path_or_hf_repo=MLX_REPO, word_timestamps=True
        )
        words = []
        for seg in out["segments"]:
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"],
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })
        return assign_speakers_max_overlap(words, self._diarize(audio_path))
