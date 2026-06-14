"""mlx-whisper (Apple-Silicon-native, large-v3) ASR + pyannote 4.x diarization +
max-overlap attribution. The speed contender on this Mac.

PROCESS ISOLATION: importing torch in the same process as Apple MLX makes
mlx_whisper segfault (native PyTorch <-> MLX/Metal conflict, verified 2026-06-14).
This adapter needs torch (for pyannote), so the mlx ASR runs in a torch-free
subprocess (eval/engines/_mlx_worker.py) and the words come back as JSON. This
module therefore must NOT import mlx_whisper at top level.

Diarization mirrors vosk_pyannote.py exactly — both adapters share the same
pyannote 4.x `token=` convention and `itertracks(yield_label=True)` pattern.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio
from pyannote.audio import Pipeline

from eval.engines.base import assign_speakers_max_overlap

# Repo root (parent of the eval/ package) so the subprocess can `-m eval...`.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class MlxWhisperPyannote:
    name = "mlxwhisper_pyannote"

    def __init__(self):
        # pyannote 4.x renamed use_auth_token -> token (mirrors vosk_pyannote.py)
        self._dia = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=os.environ["HF_TOKEN"]
        ).to(torch.device("mps"))

    def _diarize(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        out = self._dia({"waveform": waveform, "sample_rate": sr})
        # pyannote 4.x returns a DiarizeOutput whose Annotation is
        # .speaker_diarization; pyannote 3.x returned the Annotation directly.
        annotation = getattr(out, "speaker_diarization", out)
        return [
            {"speaker": spk, "start": turn.start, "end": turn.end}
            for turn, _, spk in annotation.itertracks(yield_label=True)
        ]

    def _asr(self, audio_path):
        """Run mlx-whisper in a torch-free subprocess; return word dicts."""
        proc = subprocess.run(
            [sys.executable, "-m", "eval.engines._mlx_worker", audio_path],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"mlx worker failed (exit {proc.returncode}): {proc.stderr[-2000:]}"
            )
        return json.loads(proc.stdout)

    def transcribe(self, audio_path):
        return assign_speakers_max_overlap(self._asr(audio_path), self._diarize(audio_path))
