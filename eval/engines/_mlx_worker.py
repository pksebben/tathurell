"""Subprocess worker: run mlx-whisper ASR in a torch-free process.

WHY THIS EXISTS: importing `torch` in the same process as Apple MLX makes
`mlx_whisper.transcribe` segfault (a native PyTorch <-> MLX/Metal conflict,
verified 2026-06-14). The mlx engine adapter needs torch (for pyannote
diarization), so it runs this worker as a separate process — which imports ONLY
mlx_whisper, never torch — and reads the word list back as JSON on stdout.

Usage: python -m eval.engines._mlx_worker <audio_path>
Output: JSON list of {"word","start","end"} to stdout.

Keep this module torch-free: import nothing that pulls in torch.
"""
import json
import sys

import mlx_whisper

MLX_REPO = "mlx-community/whisper-large-v3-mlx"


def transcribe_words(audio_path: str) -> list:
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
    return words


if __name__ == "__main__":
    json.dump(transcribe_words(sys.argv[1]), sys.stdout)
